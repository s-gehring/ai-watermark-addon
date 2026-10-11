export function localize(root: ParentNode = document): void {
  for (const element of root.querySelectorAll<HTMLElement>('[data-i18n]')) {
    element.textContent = browser.i18n.getMessage(element.dataset.i18n!);
  }
  document.documentElement.lang = browser.i18n.getUILanguage();
}
