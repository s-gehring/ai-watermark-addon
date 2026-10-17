"""Reads single images from remote OpenFake parquet files with HTTP range requests on the pages of the image column."""
import http.client
import struct
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from functools import lru_cache

import pyarrow.parquet as pq
from huggingface_hub import HfFileSystem, hf_hub_url

URLS, URL_LOCK, LOCAL = {}, threading.Lock(), threading.local()


class Thrift:
    def __init__(self, buf):
        self.buf, self.pos = buf, 0

    def byte(self):
        b = self.buf[self.pos]
        self.pos += 1
        return b

    def varint(self):
        shift = out = 0
        while True:
            b = self.byte()
            out |= (b & 0x7F) << shift
            if not b & 0x80:
                return out
            shift += 7

    def zigzag(self):
        n = self.varint()
        return (n >> 1) ^ -(n & 1)

    def skip(self, t):
        if t in (1, 2):
            return
        if t == 3:
            self.pos += 1
        elif t in (4, 5, 6):
            self.varint()
        elif t == 7:
            self.pos += 8
        elif t == 8:
            n = self.varint()
            self.pos += n
        elif t in (9, 10):
            h = self.byte()
            n, et = h >> 4, h & 0x0F
            if n == 15:
                n = self.varint()
            for _ in range(n):
                self.skip(et)
        elif t == 12:
            self.struct()
        else:
            raise ValueError(f"thrift type {t}")

    def struct(self):
        out, last = {}, 0
        while True:
            h = self.byte()
            if h == 0:
                return out
            delta, t = h >> 4, h & 0x0F
            fid = last + delta if delta else self.zigzag()
            last = fid
            if t in (5, 6):
                out[fid] = self.zigzag()
            elif t == 12:
                out[fid] = self.struct()
            elif t in (1, 2):
                out[fid] = t == 1
            else:
                self.skip(t)


def page_header(buf):
    r = Thrift(buf)
    h = r.struct()
    num_values = (h.get(5) or h.get(8) or {}).get(1) or (h.get(7) or {}).get(1)
    return {"type": h[1], "size": h[3], "num_values": num_values, "header_len": r.pos}


def rle_hybrid(buf, width, n):
    out, pos = [], 0
    nbytes = (width + 7) // 8
    while len(out) < n and pos < len(buf):
        header, shift = 0, 0
        while True:
            b = buf[pos]
            pos += 1
            header |= (b & 0x7F) << shift
            shift += 7
            if not b & 0x80:
                break
        if header & 1:
            groups = header >> 1
            bits = int.from_bytes(buf[pos:pos + groups * width], "little")
            pos += groups * width
            out += [(bits >> (i * width)) & ((1 << width) - 1) for i in range(groups * 8)]
        else:
            value = int.from_bytes(buf[pos:pos + nbytes], "little")
            pos += nbytes
            out += [value] * (header >> 1)
    return out[:n]


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, *args, **kwargs):
        return None


def resolve(repo, revision, filename, refresh=False):
    with URL_LOCK:
        if refresh or filename not in URLS:
            opener = urllib.request.build_opener(NoRedirect)
            location = hf_hub_url(repo, filename, repo_type="dataset", revision=revision)
            for _ in range(5):
                if urllib.parse.urlsplit(location).netloc != "huggingface.co":
                    break
                try:
                    opener.open(urllib.request.Request(location, method="HEAD"))
                    break
                except urllib.error.HTTPError as e:
                    location = urllib.parse.urljoin(location, e.headers["Location"])
            URLS[filename] = urllib.parse.urlsplit(location)
        return URLS[filename]


def connection(netloc):
    conns = LOCAL.__dict__.setdefault("conns", {})
    if netloc not in conns:
        conns[netloc] = http.client.HTTPSConnection(netloc, timeout=60)
    return conns[netloc]


class Remote:
    def __init__(self, repo, revision, filename):
        self.repo, self.revision, self.filename = repo, revision, filename
        self.url = resolve(repo, revision, filename)

    def get(self, start, length):
        for attempt in range(6):
            path = self.url.path + ("?" + self.url.query if self.url.query else "")
            conn = connection(self.url.netloc)
            try:
                conn.request("GET", path, headers={"Range": f"bytes={start}-{start + length - 1}"})
                r = conn.getresponse()
                data = r.read()
                if r.status in (200, 206):
                    return data
                if r.status in (401, 403):
                    self.url = resolve(self.repo, self.revision, self.filename, refresh=True)
                raise RuntimeError(f"HTTP {r.status}")
            except (http.client.HTTPException, OSError, RuntimeError):
                conn.close()
                LOCAL.conns.pop(self.url.netloc, None)
                time.sleep(1 + 2 * attempt)
        raise RuntimeError(f"failed {self.filename} {start}")

    def data_pages(self, offset, size):
        pos, end, row = offset, offset + size, 0
        while pos < end:
            h = None
            for length in (4096, 1 << 18, 1 << 22):
                try:
                    h = page_header(self.get(pos, length))
                    if h["header_len"] < length - 16:
                        break
                except (IndexError, ValueError, KeyError):
                    h = None
            if h is None:
                raise RuntimeError(f"bad page header {self.filename} {pos}")
            if h["type"] == 0:
                yield {"pos": pos, "header_len": h["header_len"], "size": h["size"], "first_row": row, "num_values": h["num_values"]}
                row += h["num_values"]
            pos += h["header_len"] + h["size"]

    def values(self, page, max_def):
        body = self.get(page["pos"] + page["header_len"], page["size"])
        (length,) = struct.unpack_from("<i", body, 0)
        pos, out = 4 + length, []
        for d in rle_hybrid(body[4:pos], max(1, max_def.bit_length()), page["num_values"]):
            if d == max_def:
                (n,) = struct.unpack_from("<i", body, pos)
                out.append(body[pos + 4:pos + 4 + n])
                pos += 4 + n
            else:
                out.append(None)
        return out


@lru_cache(maxsize=None)
def image_chunks(repo, revision, filename):
    meta = pq.ParquetFile(HfFileSystem().open(f"datasets/{repo}@{revision}/{filename}", block_size=1 << 15)).metadata
    column = next(j for j in range(meta.num_columns) if meta.schema.column(j).path == "image.bytes")
    max_def = meta.schema.column(column).max_definition_level
    chunks = []
    for g in range(meta.num_row_groups):
        c = meta.row_group(g).column(column)
        if c.compression != "UNCOMPRESSED" or c.has_dictionary_page:
            raise ValueError(f"{filename} row group {g}: {c.compression}, dictionary {c.has_dictionary_page}")
        chunks.append((c.data_page_offset, c.total_compressed_size, max_def))
    return chunks


def read_rows(repo, revision, filename, group, rows):
    offset, size, max_def = image_chunks(repo, revision, filename)[group]
    remote, found = Remote(repo, revision, filename), {}
    for page in remote.data_pages(offset, size):
        end = page["first_row"] + page["num_values"]
        hit = [i for i in rows if page["first_row"] <= i < end]
        if hit:
            values = remote.values(page, max_def)
            found.update((i, values[i - page["first_row"]]) for i in hit)
        if end > max(rows):
            break
    return [found.get(i) for i in rows]
