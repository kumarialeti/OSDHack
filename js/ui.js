// ui.js – Core UI logic for osdhack2026_rag project

import { initWebLLM, generateText } from "./model.js";
import { addDocument, search, loadFromDB, clearAll as clearVectorStore } from "./vectorStore.js";
import { evaluateATS } from "./ats.js";
import { showPrivacyStats, clearAllPrivacy } from "./privacy.js";
import { readFileAsync, parsePDF, markdownToText, showToast } from "./utils.js";

/**
 * Simple tab navigation
 */
function switchTab(targetId) {
  document.querySelectorAll('.tab-content').forEach(el => el.classList.remove('active'));
  document.getElementById(targetId).classList.add('active');
  // Update active button style
  document.querySelectorAll('.tab-btn').forEach(btn => btn.classList.toggle('active', btn.dataset.target === targetId));
}

// Attach tab click handlers
document.querySelectorAll('.tab-btn').forEach(btn => {
  btn.addEventListener('click', () => switchTab(btn.dataset.target));
});

// Dropdown toggle (Resources)
const dropdownToggle = document.querySelector('.dropdown-toggle');
if (dropdownToggle) {
  dropdownToggle.addEventListener('click', () => {
    const content = dropdownToggle.parentElement.querySelector('.dropdown-content');
    content.classList.toggle('hidden');
  });
}

/**
 * Model loading flow using WebLLM (Qwen2-0.5B-Instruct)
 */
let llmEngine = null;
const loadBtn = document.getElementById('load-model-btn');
const modelStatus = document.getElementById('model-status');
loadBtn.addEventListener('click', async () => {
  loadBtn.disabled = true;
  modelStatus.classList.remove('hidden');
  modelStatus.textContent = 'Downloading model...';
  try {
    llmEngine = await initWebLLM();
    modelStatus.textContent = 'Model ready!';
    showToast('Model loaded successfully', 'success');
    // Enable RAG UI now
    document.getElementById('rag-ask-btn').disabled = false;
    document.getElementById('rag-index-btn').disabled = false;
  } catch (e) {
    console.error(e);
    modelStatus.textContent = 'Failed to load model.';
    showToast('Model load failed', 'error');
    loadBtn.disabled = false;
  }
});

/**
 * RAG – file handling, indexing and answering
 */
const ragFilesInput = document.getElementById('rag-files');
const ragIndexBtn = document.getElementById('rag-index-btn');
const ragPrompt = document.getElementById('rag-prompt');
const ragAskBtn = document.getElementById('rag-ask-btn');
const ragResponse = document.getElementById('rag-response');
let pendingDocs = [];

ragFilesInput.addEventListener('change', async (e) => {
  const files = Array.from(e.target.files);
  pendingDocs = [];
  for (const f of files) {
    const content = await readFileAsync(f);
    let text = '';
    if (f.type === 'application/pdf') {
      text = await parsePDF(content);
    } else {
      const decoder = new TextDecoder('utf-8');
      text = decoder.decode(content);
    }
    pendingDocs.push({ id: f.name, text });
  }
  ragIndexBtn.disabled = false;
  showToast(`${pendingDocs.length} documents ready for indexing`, 'info');
});

ragIndexBtn.addEventListener('click', async () => {
  ragIndexBtn.disabled = true;
  const statsDiv = document.getElementById('rag-stats');
  statsDiv.classList.remove('hidden');
  statsDiv.textContent = 'Indexing...';
  await loadFromDB(); // ensure any existing store is loaded
  for (const doc of pendingDocs) {
    await addDocument(doc.id, doc.text);
  }
  statsDiv.textContent = `Indexed ${pendingDocs.length} documents.`;
  showToast('Index built', 'success');
});

ragAskBtn.addEventListener('click', async () => {
  const question = ragPrompt.value.trim();
  if (!question) return;
  ragAskBtn.disabled = true;
  ragResponse.innerHTML = '<div class="loading-container"><div class="spinner"></div>Thinking...</div>';
  try {
    const topK = await search(question, 3);
    const context = topK.map(r => r.text).join('\n');
    const prompt = `You are a helpful assistant. Answer the following question using only the provided context. If you cannot answer, say "I don't know".\n\nContext:\n${context}\n\nQuestion: ${question}`;
    const answer = await generateText(llmEngine, prompt);
    // Render answer with citations
    const citationHtml = topK.map((r, i) => `<sup>[${i + 1}]</sup>`).join(' ');
    ragResponse.innerHTML = `<p>${answer}</p><p class="citations">Sources:${citationHtml}</p>`;
  } catch (e) {
    console.error(e);
    ragResponse.textContent = 'Error during inference.';
  }
  ragAskBtn.disabled = false;
});

/**
 * ATS – resume & job description evaluation
 */
const resumeInput = document.getElementById('resume-file');
const jdInput = document.getElementById('jobdesc-file');
const atsEvalBtn = document.getElementById('ats-eval-btn');
const atsResult = document.getElementById('ats-result');
let resumeText = '';
let jdText = '';

resumeInput.addEventListener('change', async (e) => {
  const file = e.target.files[0];
  const raw = await readFileAsync(file);
  if (file.type === 'application/pdf') resumeText = await parsePDF(raw);
  else resumeText = new TextDecoder('utf-8').decode(raw);
});

jdInput.addEventListener('change', async (e) => {
  const file = e.target.files[0];
  const raw = await readFileAsync(file);
  if (file.type === 'application/pdf') jdText = await parsePDF(raw);
  else jdText = new TextDecoder('utf-8').decode(raw);
});

atsEvalBtn.addEventListener('click', async () => {
  if (!resumeText || !jdText) {
    showToast('Both resume and job description must be uploaded.', 'warning');
    return;
  }
  atsEvalBtn.disabled = true;
  atsResult.innerHTML = '<div class="loading-container"><div class="spinner"></div>Evaluating...</div>';
  try {
    const { score, missingKeywords, matchedKeywords } = evaluateATS(resumeText, jdText);
    // Render deterministic score UI
    const html = `
      <div class="ats-score-panel">
        <div class="circular-score">${score}%</div>
        <div class="ats-bars">
          <div class="ats-bar-row"><span>Matched</span><div class="bar-bg"><div class="bar-fill" style="width:${score}%"></div></div></div>
        </div>
      </div>
      <h4>Missing Keywords</h4>
      <ul>${missingKeywords.map(k => `<li>${k}</li>`).join('')}</ul>
    `;
    atsResult.innerHTML = html;
    // LLM explanation (optional, short)
    if (llmEngine) {
      const explPrompt = `Give a brief, friendly suggestion to improve the resume based on the missing keywords: ${missingKeywords.join(', ')}`;
      const advice = await generateText(llmEngine, explPrompt);
      atsResult.insertAdjacentHTML('beforeend', `<h4>LLM Advice</h4><p>${advice}</p>`);
    }
  } catch (e) {
    console.error(e);
    atsResult.textContent = 'Error during ATS evaluation.';
  }
  atsEvalBtn.disabled = false;
});

/**
 * Privacy Dashboard
 */
const privacyStatsDiv = document.getElementById('privacy-stats');
const privacyClearBtn = document.getElementById('privacy-clear-btn');

function refreshPrivacy() {
  const stats = showPrivacyStats();
  privacyStatsDiv.innerHTML = `<pre>${JSON.stringify(stats, null, 2)}</pre>`;
}

privacyClearBtn.addEventListener('click', async () => {
  await clearAllPrivacy();
  await clearVectorStore();
  refreshPrivacy();
  showToast('All local data cleared', 'success');
});

// Initial load – if model already loaded, enable UI
if (llmEngine) refreshPrivacy();
