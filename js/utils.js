// utils.js – shared helpers for the osdhack2026_rag project

/**
 * Read a File (Blob) as an ArrayBuffer.
 * @param {File} file
 * @returns {Promise<ArrayBuffer>}
 */
export function readFileAsync(file) {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => resolve(reader.result);
    reader.onerror = reject;
    reader.readAsArrayBuffer(file);
  });
}

/**
 * Parse a PDF ArrayBuffer using pdf.js and return plain text.
 * @param {ArrayBuffer} data
 * @returns {Promise<string>}
 */
export async function parsePDF(data) {
  const loadingTask = pdfjsLib.getDocument({ data });
  const pdf = await loadingTask.promise;
  let text = '';
  for (let i = 1; i <= pdf.numPages; i++) {
    const page = await pdf.getPage(i);
    const content = await page.getTextContent();
    const pageText = content.items.map(item => item.str).join(' ');
    text += pageText + '\n';
  }
  return text;
}

/**
 * Convert Markdown to plain text (strip markdown syntax).
 * Simple implementation using a regex – sufficient for our use case.
 * @param {string} md
 * @returns {string}
 */
export function markdownToText(md) {
  // Remove headings, links, images, code fences, inline code, emphasis, etc.
  return md
    .replace(/\/\*.*?\*\//gs, '') // block comments
    .replace(/`{3}[\s\S]*?`{3}/g, '') // code fences
    .replace(/`([^`]+)`/g, '$1') // inline code
    .replace(/!\[.*?\]\(.*?\)/g, '') // images
    .replace(/\[([^\]]+)\]\([^\)]+\)/g, '$1') // links
    .replace(/[#>*_-]+/g, '') // markdown symbols
    .replace(/\n{2,}/g, '\n') // collapse blank lines
    .trim();
}

/**
 * Show a toast notification.
 * @param {string} message
 * @param {'success'|'error'|'info'|'warning'} type
 */
export function showToast(message, type = 'info') {
  const toast = document.createElement('div');
  toast.className = `toast toast-${type}`;
  toast.textContent = message;
  document.body.appendChild(toast);
  setTimeout(() => toast.remove(), 3000);
}

/**
 * Compute cosine similarity between two Float32Array vectors.
 * @param {Float32Array} a
 * @param {Float32Array} b
 * @returns {number} similarity in [0,1]
 */
export function cosineSimilarity(a, b) {
  let dot = 0,
    normA = 0,
    normB = 0;
  for (let i = 0; i < a.length; i++) {
    dot += a[i] * b[i];
    normA += a[i] * a[i];
    normB += b[i] * b[i];
  }
  return dot / (Math.sqrt(normA) * Math.sqrt(normB) + 1e-9);
}
