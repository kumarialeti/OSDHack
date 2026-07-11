// privacy.js – dashboard for local storage stats and clear‑all functionality

import { db } from "./vectorStore.js";

/**
 * Calculate approximate size (in bytes) of a stringified JSON object.
 */
function approxSizeBytes(obj) {
  return new Blob([JSON.stringify(obj)], { type: 'application/json' }).size;
}

/**
 * Retrieve statistics about stored documents and embeddings.
 * Returns an object that can be displayed in the UI.
 */
export async function showPrivacyStats() {
  const docs = await db.docs.toArray();
  const docCount = docs.length;
  // Compute total size of embeddings (stored as JSON strings)
  const embeddingsSize = docs.reduce((acc, d) => acc + approxSizeBytes(d.embedding), 0);
  const docsSize = docs.reduce((acc, d) => acc + approxSizeBytes(d.text), 0);
  const totalDBSizeKB = Math.round((embeddingsSize + docsSize) / 1024);

  // Model cache size – WebLLM stores model files in Cache Storage under the "webllm" cache name.
  let modelCacheSizeKB = 0;
  try {
    const cache = await caches.open('webllm');
    const requests = await cache.keys();
    for (const req of requests) {
      const resp = await cache.match(req);
      if (resp) {
        const clone = resp.clone();
        const blob = await clone.blob();
        modelCacheSizeKB += Math.round(blob.size / 1024);
      }
    }
  } catch (e) {
    // If the cache does not exist, ignore.
    console.warn('WebLLM cache not found:', e);
  }

  return {
    documents: docCount,
    totalEmbeddingSizeKB: Math.round(embeddingsSize / 1024),
    totalDocumentTextSizeKB: Math.round(docsSize / 1024),
    modelCacheSizeKB,
    totalLocalStorageKB: totalDBSizeKB + modelCacheSizeKB,
  };
}

/**
 * Clear all IndexedDB data and model cache.
 */
export async function clearAllPrivacy() {
  // Clear Dexie DB
  await db.docs.clear();
  // Clear WebLLM model cache (if any)
  try {
    const cacheNames = await caches.keys();
    for (const name of cacheNames) {
      if (name.includes('webllm')) {
        await caches.delete(name);
      }
    }
  } catch (e) {
    console.warn('Error clearing WebLLM caches:', e);
  }
}
