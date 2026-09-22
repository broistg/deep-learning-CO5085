/**
 * API Client Helper for Course Demo
 */

const API_BASE = import.meta.env.VITE_API_BASE_URL || "";

export function apiUrl(path) {
  return `${API_BASE}${path}`;
}

export async function fetchJson(url, options = {}) {
  const res = await fetch(url, options);
  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error(err.error || `HTTP ${res.status}: ${res.statusText}`);
  }
  return res.json();
}

/**
 * Example prediction function for image classification
 * @param {File} file
 * @param {string} modelId
 */
export async function predictImage(file, modelId) {
  const fd = new FormData();
  fd.append("file", file);
  fd.append("model", modelId);

  return fetchJson(apiUrl("/api/predict/image"), {
    method: "POST",
    body: fd,
  });
}

/**
 * Example prediction function for text classification
 * @param {string} text
 * @param {string} modelId
 */
export async function predictText(text, modelId) {
  return fetchJson(apiUrl("/api/predict/text"), {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ text, model: modelId }),
  });
}
