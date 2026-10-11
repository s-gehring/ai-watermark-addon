import { localize } from '../shared/i18n';
import { loadSettings, saveSettings, type Settings } from '../shared/settings';

const form = document.getElementById('form') as HTMLFormElement;
const field = <T extends HTMLElement>(name: string) => form.elements.namedItem(name) as unknown as T;
const thresholdValue = document.getElementById('thresholdValue') as HTMLOutputElement;
const saved = document.getElementById('saved')!;
let savedTimer = 0;

function render(settings: Settings): void {
  field<HTMLInputElement>('minSize').value = String(settings.minSize);
  field<HTMLInputElement>('threshold').value = String(Math.round(settings.threshold * 100));
  thresholdValue.value = `${Math.round(settings.threshold * 100)} %`;
  for (const key of ['checkMetadata', 'includeBackgrounds', 'showScore', 'showChecked'] as const) {
    field<HTMLInputElement>(key).checked = settings[key];
  }
  field<HTMLTextAreaElement>('disabledHosts').value = settings.disabledHosts.join('\n');
}

async function save(): Promise<void> {
  const next = await saveSettings({
    minSize: Number(field<HTMLInputElement>('minSize').value),
    threshold: Number(field<HTMLInputElement>('threshold').value) / 100,
    checkMetadata: field<HTMLInputElement>('checkMetadata').checked,
    includeBackgrounds: field<HTMLInputElement>('includeBackgrounds').checked,
    showScore: field<HTMLInputElement>('showScore').checked,
    showChecked: field<HTMLInputElement>('showChecked').checked,
    disabledHosts: field<HTMLTextAreaElement>('disabledHosts').value.split(/\s+/).filter(Boolean),
  });
  thresholdValue.value = `${Math.round(next.threshold * 100)} %`;
  saved.hidden = false;
  clearTimeout(savedTimer);
  savedTimer = window.setTimeout(() => (saved.hidden = true), 1500);
}

localize();
void loadSettings().then(render);
form.addEventListener('input', (event) => {
  if ((event.target as HTMLInputElement).name === 'threshold') {
    thresholdValue.value = `${field<HTMLInputElement>('threshold').value} %`;
  }
});
form.addEventListener('change', () => void save());
