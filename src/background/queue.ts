export async function firstResult<T>(candidates: string[], load: (candidate: string) => Promise<T | null>): Promise<T | null> {
  for (const candidate of candidates) {
    const result = await load(candidate);
    if (result) return result;
  }
  return null;
}
