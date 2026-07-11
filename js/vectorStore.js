// vectorStore.js – simple in‑browser vector store using ONNX Runtime Web embeddings

import Dexie from "https://cdn.jsdelivr.net/npm/dexie@3.2.2/dist/dexie.mjs";
import { readFileAsync } from "./utils.js";
import * as ort from "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.15.1/dist/ort.min.js";

// ---- Dexie DB schema ----
const db = new Dexie('LocalAIVectorDB');
// Store: ++id (auto), docId (string), text (string), embedding (array of numbers serialized as JSON)
db.version(1).stores({
  docs: '++id, docId',
});

// In‑memory cache of vectors for fast search
let vectorCache = [];

let embeddingSession = null;
const EMBEDDING_MODEL_URL = './models/all-MiniLM-L6-v2.onnx'; // place model file here

/** Load ONNX Runtime embedding model lazily */
async function loadEmbeddingModel() {
  if (embeddingSession) return embeddingSession;
  // Ensure ort is ready
  await ort.env.wasm.wasmPath = 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.15.1/dist/';
  embeddingSession = await ort.InferenceSession.create(EMBEDDING_MODEL_URL);
  return embeddingSession;
}

/** Simple tokenization: split on whitespace, map to ids via a tiny vocab (fallback) */
function simpleTokenizer(text) {
  // Very naive: each word gets an incremental id (mod 10000)
  const words = text.toLowerCase().split(/\s+/).filter(Boolean);
  const ids = words.map(w => (w.charCodeAt(0) * w.length) % 10000);
  // Pad / truncate to max length 128
  const maxLen = 128;
  const inputIds = new Uint32Array(maxLen);
  inputIds.fill(0);
  ids.slice(0, maxLen).forEach((v,i) => inputIds[i] = v);
  // attention mask
  const attMask = new Uint32Array(maxLen);
  attMask.fill(0);
  ids.slice(0, maxLen).forEach((_,i) => attMask[i] = 1);
  return { inputIds, attMask };
}

/** Compute embedding for a piece of text */
export async function getEmbedding(text) {
  const session = await loadEmbeddingModel();
  const { inputIds, attMask } = simpleTokenizer(text);
  const feeds = {
    input_ids: new ort.Tensor('int64', inputIds, [1, inputIds.length]),
    attention_mask: new ort.Tensor('int64', attMask, [1, attMask.length]),
  };
  const results = await session.run(feeds);
  // Assume output name is 'last_hidden_state' and we take mean pooling
  const hidden = results['last_hidden_state']; // shape [1, seq_len, hidden]
  const data = hidden.data;
  const hiddenSize = hidden.dims[2];
  const seqLen = hidden.dims[1];
  const avg = new Float32Array(hiddenSize);
  for (let i = 0; i < seqLen; i++) {
    for (let j = 0; j < hiddenSize; j++) {
      avg[j] += data[i * hiddenSize + j];
    }
  }
  for (let j = 0; j < hiddenSize; j++) avg[j] /= seqLen;
  return avg;
}

/** Add a document to the store */
export async function addDocument(docId, text) {
  const embedding = await getEmbedding(text);
  const doc = { docId, text, embedding: JSON.stringify(Array.from(embedding)) };
  const id = await db.docs.add(doc);
  // Keep in‑memory cache up‑to‑date
  vectorCache.push({ id, docId, text, embedding });
  return id;
}

/** Load all stored vectors into memory */
export async function loadFromDB() {
  const all = await db.docs.toArray();
  vectorCache = all.map(d => ({
    id: d.id,
    docId: d.docId,
    text: d.text,
    embedding: Float32Array.from(JSON.parse(d.embedding))
  }));
}

/** Search for top‑k most similar documents */
export async function search(query, k = 3) {
  const queryEmb = await getEmbedding(query);
  // Compute similarity against cached vectors
  const scores = vectorCache.map(v => ({
    id: v.id,
    docId: v.docId,
    text: v.text,
    score: cosineSimilarity(queryEmb, v.embedding)
  }));
  scores.sort((a,b) => b.score - a.score);
  return scores.slice(0, k);
}

/** Clear all stored documents */
export async function clearAll() {
  await db.docs.clear();
  vectorCache = [];
}

/** Helper cosine similarity (re‑using utils version) */
function cosineSimilarity(a, b) {
  let dot = 0, normA = 0, normB = 0;
  for (let i = 0; i < a.length; i++) {
    dot += a[i] * b[i];
    normA += a[i] * a[i];
    normB += b[i] * b[i];
  }
  return dot / (Math.sqrt(normA) * Math.sqrt(normB) + 1e-9);
}

/** Export DB handle for other modules if needed */
export { db };
