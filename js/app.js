/**
 * app.js – Single entry-point for OSDHack 2026 On‑Device AI
 *
 * Architecture:
 *   Phase 1 – WebLLM model loading (Qwen2-0.5B-Instruct)
 *   Phase 2 – File upload & text extraction (PDF / TXT / MD)
 *   Phase 3 – Chunking & ONNX embedding (all-MiniLM-L6-v2 via ort)
 *   Phase 4 – IndexedDB storage via Dexie
 *   Phase 5+6 – RAG retrieval + WebLLM answer generation
 *   Phase 7 – ATS evaluation (deterministic + LLM advice)
 *   Phase 8 – Privacy dashboard
 */

// ─────────────────────────────────────────────
// IMPORTS (CDN ESM)
// ─────────────────────────────────────────────
import Dexie from 'https://cdn.jsdelivr.net/npm/dexie@3.2.7/dist/dexie.mjs';
import { pipeline, env } from 'https://cdn.jsdelivr.net/npm/@xenova/transformers@2.17.2';

// Configure Transformers.js environment
env.allowLocalModels = false;
env.backends.onnx.wasm.numThreads = 1;

// ─────────────────────────────────────────────
// CONSTANTS
// ─────────────────────────────────────────────
const CHUNK_SIZE  = 600;   // characters per chunk
const CHUNK_OVER  = 100;   // overlap characters
const TOP_K       = 3;     // chunks to retrieve

// Embedding dimension for all-MiniLM-L6-v2 = 384
// We will use a TF-IDF-style sparse cosine as fallback if ONNX fails
const EMBEDDING_DIM = 384;

// ─────────────────────────────────────────────
// DATABASE (Dexie / IndexedDB)
// ─────────────────────────────────────────────
const db = new Dexie('OSDHack2026');
db.version(1).stores({
  chunks: '++id, docId, docName',
  // embedding stored as JSON string
});

// In-memory vector cache for fast search
let vectorCache = [];   // [{id, docId, docName, text, vec: Float32Array}]

async function dbLoadAll() {
  const rows = await db.chunks.toArray();
  vectorCache = rows.map(r => ({
    id:      r.id,
    docId:   r.docId,
    docName: r.docName,
    text:    r.text,
    vec:     Float32Array.from(JSON.parse(r.vec)),
  }));
}

async function dbAddChunk(docId, docName, text, vec) {
  const id = await db.chunks.add({ docId, docName, text, vec: JSON.stringify(Array.from(vec)) });
  vectorCache.push({ id, docId, docName, text, vec });
  return id;
}

async function dbClear() {
  await db.chunks.clear();
  vectorCache = [];
}

// ─────────────────────────────────────────────
// EMBEDDINGS (Transformers.js – all-MiniLM-L6-v2)
// Uses proper tokenization and semantic pooling
// ─────────────────────────────────────────────
let extractor = null;
let embeddingReady = false;

async function loadEmbeddingModel() {
  try {
    // Load Transformers.js feature-extraction pipeline
    extractor = await pipeline('feature-extraction', 'Xenova/all-MiniLM-L6-v2');
    embeddingReady = true;
    toast('Embedding model loaded ✓', 'success');
    console.log('[Embedding] Transformers.js all-MiniLM-L6-v2 ready');
  } catch (e) {
    console.warn('[Embedding] Failed to load Transformers.js pipeline, using fallback:', e);
    embeddingReady = false;
  }
}

/**
 * Compute an embedding for a text chunk.
 * Primary: Transformers.js all-MiniLM-L6-v2
 * Fallback: simple TF-IDF character n-gram vector (normalized)
 */
async function embed(text) {
  if (embeddingReady && extractor) {
    return await transformerEmbed(text);
  }
  return tfidfEmbed(text);
}

async function transformerEmbed(text) {
  try {
    const output = await extractor(text, { pooling: 'mean', normalize: true });
    return output.data; // Float32Array
  } catch (e) {
    console.warn('[Embedding] transformerEmbed error, using fallback:', e);
    return tfidfEmbed(text);
  }
}

/** Simple TF-IDF-inspired character bigram embedding (dim=384) */
function tfidfEmbed(text) {
  const vec = new Float32Array(EMBEDDING_DIM);
  const clean = text.toLowerCase().replace(/[^a-z0-9 ]/g, '');
  for (let i = 0; i < clean.length - 1; i++) {
    const a = clean.charCodeAt(i);
    const b = clean.charCodeAt(i + 1);
    const idx = ((a * 31 + b) % EMBEDDING_DIM + EMBEDDING_DIM) % EMBEDDING_DIM;
    vec[idx] += 1;
  }
  return normalise(vec);
}

function normalise(vec) {
  let norm = 0;
  for (let i = 0; i < vec.length; i++) norm += vec[i] * vec[i];
  norm = Math.sqrt(norm) || 1;
  for (let i = 0; i < vec.length; i++) vec[i] /= norm;
  return vec;
}

function cosine(a, b) {
  let dot = 0;
  for (let i = 0; i < a.length; i++) dot += a[i] * b[i];
  return dot;
}

// ─────────────────────────────────────────────
// CHUNKING
// ─────────────────────────────────────────────
function chunkText(text) {
  const chunks = [];
  let start = 0;
  while (start < text.length) {
    const end = Math.min(start + CHUNK_SIZE, text.length);
    chunks.push(text.slice(start, end).trim());
    start += CHUNK_SIZE - CHUNK_OVER;
  }
  return chunks.filter(c => c.length > 20);
}

// ─────────────────────────────────────────────
// FILE PARSING (PDF / TXT / MD)
// ─────────────────────────────────────────────
async function parseFile(file) {
  try {
    const arrayBuffer = await file.arrayBuffer();

    if (file.name.toLowerCase().endsWith('.pdf')) {
      if (typeof pdfjsLib === 'undefined') {
        throw new Error('PDF.js not loaded – include the script tag in index.html');
      }
      const pdf = await pdfjsLib.getDocument({ data: new Uint8Array(arrayBuffer) }).promise;
      let text = '';
      for (let i = 1; i <= pdf.numPages; i++) {
        const page    = await pdf.getPage(i);
        const content = await page.getTextContent();
        // Join with space; keep newlines between pages
        text += content.items.map(it => it.str).join(' ') + '\n';
      }
      const trimmed = text.trim();
      console.log(`[Parse] PDF "${file.name}" – ${pdf.numPages} pages, ${trimmed.length} chars extracted`);
      if (trimmed.length < 20) {
        toast(`⚠️ PDF "${file.name}" appears to be image-only (no extractable text)`, 'warning');
      }
      return trimmed;
    }

    // TXT / MD
    const text = new TextDecoder('utf-8').decode(arrayBuffer);
    console.log(`[Parse] Text file "${file.name}" – ${text.length} chars`);
    return text;
  } catch (e) {
    console.error('[Parse] Failed to parse file:', file.name, e);
    toast(`Failed to parse "${file.name}": ${e.message}`, 'error');
    return '';
  }
}

// ─────────────────────────────────────────────
// WEBLLM (Qwen2-0.5B-Instruct)
// ─────────────────────────────────────────────
let llmEngine = null;
let llmReady  = false;

async function loadWebLLM(onProgress) {
  // WebLLM from CDN
  const { CreateMLCEngine } = await import(
    'https://esm.run/@mlc-ai/web-llm'
  );
  llmEngine = await CreateMLCEngine(
    'Qwen2-0.5B-Instruct-q4f16_1-MLC',
    {
      initProgressCallback: (p) => {
        const pct = Math.round((p.progress ?? 0) * 100);
        onProgress(pct, p.text ?? 'Loading…');
      },
    }
  );
  llmReady = true;
}

async function llmGenerate(prompt, maxTokens = 300) {
  if (!llmReady || !llmEngine) throw new Error('Model not loaded');
  const reply = await llmEngine.chat.completions.create({
    messages: [{ role: 'user', content: prompt }],
    max_tokens: maxTokens,
    temperature: 0.6,
  });
  return reply.choices[0].message.content.trim();
}

// ─────────────────────────────────────────────
// ATS SCORING (deterministic)
// ─────────────────────────────────────────────
const STOP = new Set(['the','and','or','a','an','to','of','in','for','with','on','at','by','as','is',
  'are','was','were','be','been','will','should','would','could','can','do','does','did','has',
  'have','had','this','that','these','those','their','your','our','its','not','but','if','then']);

const RESUME_SECTIONS = ['experience','education','skills','projects','summary','objective','certifications'];

function extractKeywords(text) {
  return [...new Set(
    text.toLowerCase().split(/[^a-z0-9+#]+/).filter(w => w.length > 2 && !STOP.has(w))
  )];
}

function atsScore(resumeText, jdText) {
  const jdKws = extractKeywords(jdText);
  const resumeWords = new Set(extractKeywords(resumeText));

  const matched = jdKws.filter(k => resumeWords.has(k));
  const missing  = jdKws.filter(k => !resumeWords.has(k));
  const keywordPct = Math.round((matched.length / Math.max(jdKws.length, 1)) * 100);

  // Skills: look for technical keywords (length > 3, contain digit or uppercase indicator)
  const techKws = jdKws.filter(k => k.length > 3);
  const matchedTech = techKws.filter(k => resumeWords.has(k));
  const skillsPct = Math.round((matchedTech.length / Math.max(techKws.length, 1)) * 100);

  // Section completeness
  const lower = resumeText.toLowerCase();
  const foundSections = RESUME_SECTIONS.filter(s => lower.includes(s));
  const sectionPct = Math.round((foundSections.length / RESUME_SECTIONS.length) * 100);

  const overall = Math.round((keywordPct * 0.5) + (skillsPct * 0.3) + (sectionPct * 0.2));
  return { overall, keywordPct, skillsPct, sectionPct, matched, missing };
}

// ─────────────────────────────────────────────
// TOAST NOTIFICATIONS
// ─────────────────────────────────────────────
function toast(msg, type = 'info') {
  const el = document.createElement('div');
  el.className = `toast toast-${type}`;
  el.textContent = msg;
  document.getElementById('toast-container').appendChild(el);
  setTimeout(() => el.remove(), 3000);
}

// ─────────────────────────────────────────────
// PIPELINE STEP HIGHLIGHTS
// ─────────────────────────────────────────────
function activatePipe(stepId) {
  document.querySelectorAll('.pipeline-step').forEach(el => el.classList.remove('active'));
  const el = document.getElementById(stepId);
  if (el) el.classList.add('active');
}

// ─────────────────────────────────────────────
// ─────── UI WIRING ──────────────────────────
// ─────────────────────────────────────────────

// ─── TAB NAVIGATION ───
document.querySelectorAll('.nav-btn').forEach(btn => {
  btn.addEventListener('click', () => {
    document.querySelectorAll('.nav-btn').forEach(b => b.classList.remove('active'));
    document.querySelectorAll('.tab-pane').forEach(p => p.classList.remove('active'));
    btn.classList.add('active');
    document.getElementById('tab-' + btn.dataset.tab).classList.add('active');
    if (btn.dataset.tab === 'privacy') refreshPrivacy();
  });
});

// ─── PHASE 1: MODEL LOADING ───
const loadBtn      = document.getElementById('load-model-btn');
const statusDot    = document.getElementById('status-dot');
const statusText   = document.getElementById('model-status-text');
const progressWrap = document.getElementById('model-progress-wrap');
const progressFill = document.getElementById('model-progress-fill');
const progressLbl  = document.getElementById('model-progress-label');
const modelBadge   = document.getElementById('model-badge');

loadBtn.addEventListener('click', async () => {
  loadBtn.disabled = true;
  statusDot.className = 'status-dot loading';
  statusText.textContent = 'Downloading Qwen2-0.5B-Instruct…';
  progressWrap.classList.remove('hidden');

  // Start embedding model load in parallel
  loadEmbeddingModel();

  try {
    await loadWebLLM((pct, text) => {
      progressFill.style.width = pct + '%';
      progressLbl.textContent  = pct + '%';
      statusText.textContent   = text;
    });
    statusDot.className = 'status-dot ready';
    statusText.textContent = 'Qwen2-0.5B-Instruct ready';
    progressWrap.classList.add('hidden');
    modelBadge.textContent = 'Model: Qwen2-0.5B ✓';
    modelBadge.style.color = 'var(--green)';
    toast('Model loaded! Ready for inference.', 'success');
    activatePipe('pipe-llm');
  } catch (e) {
    statusDot.className = 'status-dot error';
    statusText.textContent = 'Error: ' + (e.message || 'failed');
    toast('Model load failed: ' + e.message, 'error');
    loadBtn.disabled = false;
    console.error(e);
  }
});

// ─── PHASE 2: FILE UPLOAD (RAG) ───
const ragDropZone  = document.getElementById('rag-drop-zone');
const ragFileInput = document.getElementById('rag-file-input');
const ragBrowseBtn = document.getElementById('rag-browse-btn');
const ragDocList   = document.getElementById('rag-doc-list');
const ragIndexActs = document.getElementById('rag-index-actions');
const ragIndexBtn  = document.getElementById('rag-index-btn');
const ragIndexProg = document.getElementById('rag-index-progress');
const ragIndexFill = document.getElementById('rag-index-fill');
const ragIndexLbl  = document.getElementById('rag-index-label');
const ragStats     = document.getElementById('rag-stats');

let pendingFiles = [];

ragBrowseBtn.addEventListener('click', () => ragFileInput.click());
ragFileInput.addEventListener('change', () => handleFileList(Array.from(ragFileInput.files)));
ragDropZone.addEventListener('dragover', e => { e.preventDefault(); ragDropZone.classList.add('dragover'); });
ragDropZone.addEventListener('dragleave', () => ragDropZone.classList.remove('dragover'));
ragDropZone.addEventListener('drop', e => {
  e.preventDefault();
  ragDropZone.classList.remove('dragover');
  handleFileList(Array.from(e.dataTransfer.files));
});
ragDropZone.addEventListener('click', (e) => {
  if (e.target !== ragBrowseBtn) ragFileInput.click();
});

function handleFileList(files) {
  pendingFiles = files.filter(f => f.name.match(/\.(pdf|txt|md)$/i));
  ragDocList.innerHTML = '';
  for (const f of pendingFiles) {
    const li = document.createElement('li');
    li.textContent = f.name + ' (' + (f.size / 1024).toFixed(1) + ' KB)';
    ragDocList.appendChild(li);
  }
  if (pendingFiles.length > 0) {
    ragIndexActs.style.display = 'flex';
    activatePipe('pipe-upload');
    toast(`${pendingFiles.length} file(s) selected`, 'info');
  }
}

// ─── PHASE 3+4: INDEXING ───
ragIndexBtn.addEventListener('click', async () => {
  if (pendingFiles.length === 0) return;
  ragIndexBtn.disabled = true;
  ragIndexProg.classList.remove('hidden');
  ragIndexFill.style.width = '0%';

  let totalChunks = 0;
  let processed   = 0;

  // First pass: parse + chunk all files
  const parsed = [];
  for (const f of pendingFiles) {
    ragIndexLbl.textContent = `Parsing "${f.name}"…`;
    activatePipe('pipe-chunk');
    const text   = await parseFile(f);
    const chunks = chunkText(text);
    console.log(`[Index] "${f.name}" → ${chunks.length} chunks`);
    if (chunks.length === 0) {
      toast(`⚠️ No text found in "${f.name}" – skipping`, 'warning');
      continue;
    }
    parsed.push({ name: f.name, chunks });
    totalChunks += chunks.length;
  }

  if (totalChunks === 0) {
    ragIndexLbl.textContent = 'No indexable text found in uploaded files.';
    toast('No text could be extracted. Use text-based PDFs or TXT/MD files.', 'error');
    ragIndexBtn.disabled = false;
    return;
  }

  // Second pass: embed and store
  activatePipe('pipe-embed');
  for (const { name, chunks } of parsed) {
    const docId = name + '_' + Date.now();
    for (const chunk of chunks) {
      const vec = await embed(chunk);
      await dbAddChunk(docId, name, chunk, vec);
      processed++;
      const pct = Math.round((processed / totalChunks) * 100);
      ragIndexFill.style.width  = pct + '%';
      ragIndexLbl.textContent   = `Embedding ${processed}/${totalChunks} chunks…`;
    }
  }

  activatePipe('pipe-store');
  ragIndexProg.classList.add('hidden');
  ragStats.classList.remove('hidden');

  // Update stat chips
  const uniqueDocs = new Set(vectorCache.map(v => v.docId)).size;
  document.getElementById('stat-docs').textContent    = uniqueDocs + ' doc(s)';
  document.getElementById('stat-chunks').textContent  = vectorCache.length + ' chunks';
  document.getElementById('stat-vectors').textContent = vectorCache.length + ' vectors';

  console.log(`[Index] Done – ${vectorCache.length} vectors in cache`);
  toast(`✅ Indexed ${totalChunks} chunks from ${parsed.length} file(s). Ready to answer questions!`, 'success');
  ragIndexBtn.disabled = false;
});

// ─── PHASE 5+6: RAG QUERY ───
const ragQuestion  = document.getElementById('rag-question');
const ragAskBtn    = document.getElementById('rag-ask-btn');
const ragAnswerBox = document.getElementById('rag-answer-box');
const ragThinking  = document.getElementById('rag-thinking');
const ragAnswerTxt = document.getElementById('rag-answer-text');
const ragCitations = document.getElementById('rag-citations');
const ragActionsEl = document.getElementById('rag-answer-actions');
const ragCopyBtn   = document.getElementById('rag-copy-btn');
const ragDlBtn     = document.getElementById('rag-download-btn');

ragAskBtn.addEventListener('click', async () => {
  const q = ragQuestion.value.trim();
  if (!q) { toast('Please enter a question', 'warning'); return; }

  // Show answer box immediately so user sees progress
  ragAnswerBox.classList.remove('hidden');
  ragThinking.classList.remove('hidden');
  ragAnswerTxt.textContent = '';
  ragCitations.classList.add('hidden');
  ragActionsEl.classList.add('hidden');
  ragAskBtn.disabled = true;

  try {
    // Guard: no indexed docs?
    if (vectorCache.length === 0) {
      ragThinking.classList.add('hidden');
      ragAnswerTxt.textContent = '⚠️ No documents indexed yet.\n\nSteps:\n1. Upload PDF/TXT/MD files\n2. Click "Build Index"\n3. Then ask your question.';
      ragAskBtn.disabled = false;
      return;
    }

    // Phase 5: Retrieve top-K chunks
    activatePipe('pipe-retrieve');
    const queryVec = await embed(q);
    const scored = vectorCache
      .map(v => ({ ...v, score: cosine(queryVec, v.vec) }))
      .sort((a, b) => b.score - a.score)
      .slice(0, TOP_K);

    console.log('[RAG] Top chunks:', scored.map(s => ({ score: s.score.toFixed(3), preview: s.text.slice(0,60) })));
    const context = scored.map((s, i) => `[${i + 1}] ${s.text}`).join('\n\n');

    // Phase 6: Generate answer
    activatePipe('pipe-llm');
    let answer;
    if (llmReady) {
      const prompt =
        `You are a helpful assistant. Answer the question using ONLY the context below.\n` +
        `If the answer is not in the context, say "I don't know based on the provided documents."\n\n` +
        `Context:\n${context}\n\nQuestion: ${q}\n\nAnswer:`;
      answer = await llmGenerate(prompt);
    } else {
      // No LLM loaded – show retrieved chunks directly (still useful!)
      answer =
        `ℹ️ AI model not loaded yet (click "Load Model" on the Home tab for full answers).\n\n` +
        `📄 Most relevant passage from your documents:\n\n${scored[0]?.text ?? 'Nothing found.'}`;
    }

    activatePipe('pipe-answer');
    ragThinking.classList.add('hidden');
    ragAnswerTxt.textContent = answer;

    // Citations
    ragCitations.classList.remove('hidden');
    ragCitations.innerHTML = '<strong>Sources:</strong> ' +
      scored.map((s, i) => `<span class="citation-chip">[${i + 1}] ${s.docName} (score: ${s.score.toFixed(2)})</span>`).join('');

    ragActionsEl.classList.remove('hidden');
  } catch (e) {
    ragThinking.classList.add('hidden');
    ragAnswerTxt.textContent = '❌ Error: ' + (e.message || 'inference failed');
    toast('Inference error: ' + e.message, 'error');
    console.error('[RAG] Error:', e);
  }
  ragAskBtn.disabled = false;
});

// Copy / Download
ragCopyBtn.addEventListener('click', () => {
  navigator.clipboard.writeText(ragAnswerTxt.textContent);
  toast('Copied!', 'success');
});
ragDlBtn.addEventListener('click', () => {
  const blob = new Blob([ragAnswerTxt.textContent], { type: 'text/plain' });
  const a = document.createElement('a');
  a.href = URL.createObjectURL(blob);
  a.download = 'answer.txt';
  a.click();
});

// ─── PHASE 7: ATS ───
let resumeText = '';
let jdText     = '';

function setupDropZone(dropId, inputId, labelId, onText) {
  const zone  = document.getElementById(dropId);
  const input = document.getElementById(inputId);
  const label = document.getElementById(labelId);

  zone.addEventListener('click', () => input.click());
  zone.addEventListener('dragover', e => { e.preventDefault(); zone.classList.add('dragover'); });
  zone.addEventListener('dragleave', () => zone.classList.remove('dragover'));
  zone.addEventListener('drop', async e => {
    e.preventDefault(); zone.classList.remove('dragover');
    const file = e.dataTransfer.files[0];
    if (file) { label.textContent = '✅ ' + file.name; onText(await parseFile(file)); checkATS(); }
  });
  input.addEventListener('change', async () => {
    const file = input.files[0];
    if (file) { label.textContent = '✅ ' + file.name; onText(await parseFile(file)); checkATS(); }
  });
}

setupDropZone('resume-drop-zone', 'resume-input', 'resume-label', t => { resumeText = t; });
setupDropZone('jd-drop-zone',     'jd-input',     'jd-label',     t => { jdText     = t; });

const atsEvalBtn    = document.getElementById('ats-eval-btn');
const atsResultCard = document.getElementById('ats-result-card');

function checkATS() {
  atsEvalBtn.disabled = !(resumeText && jdText);
}

atsEvalBtn.addEventListener('click', async () => {
  atsEvalBtn.disabled = true;
  const { overall, keywordPct, skillsPct, sectionPct, matched, missing } = atsScore(resumeText, jdText);

  // Show card
  atsResultCard.classList.remove('hidden');

  // Animate circular score
  const scoreEl   = document.getElementById('ats-score-number');
  const ringFill  = document.getElementById('score-ring-fill');
  const circumference = 2 * Math.PI * 50; // r=50
  scoreEl.textContent = overall;
  ringFill.style.strokeDashoffset = circumference - (circumference * overall / 100);

  // Mini bars
  const setBar = (id, pctId, pct) => {
    document.getElementById(id).style.width   = pct + '%';
    document.getElementById(pctId).textContent = pct + '%';
  };
  setBar('bar-keywords', 'bar-keywords-pct', keywordPct);
  setBar('bar-skills',   'bar-skills-pct',   skillsPct);
  setBar('bar-sections', 'bar-sections-pct', sectionPct);

  // Keyword tags
  const matchedEl = document.getElementById('kw-matched');
  const missingEl = document.getElementById('kw-missing');
  matchedEl.innerHTML = matched.slice(0, 30).map(k => `<span class="kw-tag kw-tag-matched">${k}</span>`).join('');
  missingEl.innerHTML = missing.slice(0, 30).map(k => `<span class="kw-tag kw-tag-missing">${k}</span>`).join('');

  // LLM advice
  const adviceEl = document.getElementById('ats-advice-text');
  const spinner  = document.getElementById('advice-spinner');
  if (llmReady) {
    adviceEl.textContent = '';
    spinner.classList.remove('hidden');
    try {
      const prompt = `You are an expert resume coach. The candidate is missing these important keywords from their resume: ${missing.slice(0, 15).join(', ')}. Give 3 specific, actionable suggestions to improve their resume. Be concise.`;
      const advice = await llmGenerate(prompt, 250);
      adviceEl.textContent = advice;
    } catch (e) {
      adviceEl.textContent = 'Could not generate advice: ' + e.message;
    }
    spinner.classList.add('hidden');
  } else {
    adviceEl.textContent = 'Load the AI model on the Home tab to get personalized improvement suggestions.';
  }

  toast(`ATS Score: ${overall}%`, overall >= 70 ? 'success' : overall >= 40 ? 'warning' : 'error');
  atsEvalBtn.disabled = false;
});

// ─── PHASE 8: PRIVACY DASHBOARD ───
async function refreshPrivacy() {
  const rows = await db.chunks.toArray();
  const uniqueDocs = new Set(rows.map(r => r.docId)).size;
  document.getElementById('prv-docs').textContent   = uniqueDocs;
  document.getElementById('prv-chunks').textContent = rows.length;
  // Approximate size
  const sizeBytes = rows.reduce((a, r) => a + (r.text?.length ?? 0) + (r.vec?.length ?? 0), 0);
  document.getElementById('prv-size').textContent  = Math.round(sizeBytes / 1024);
  document.getElementById('prv-model').textContent = llmReady ? 'Loaded ✓' : 'Not loaded';
}

document.getElementById('privacy-refresh-btn').addEventListener('click', refreshPrivacy);
document.getElementById('privacy-clear-btn').addEventListener('click', async () => {
  await dbClear();
  await refreshPrivacy();
  // Also clear model cache if possible
  try {
    const keys = await caches.keys();
    for (const k of keys) { if (k.includes('webllm') || k.includes('mlc')) await caches.delete(k); }
  } catch(_) {}
  toast('All local data cleared', 'success');
});

// ─── INIT ───
(async () => {
  await dbLoadAll();
  toast('App ready – load a model to begin', 'info');
})();
