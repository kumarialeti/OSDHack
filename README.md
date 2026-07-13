# OSDHack 2026 – On‑Device AI Knowledge Base

### Demo

- Video demo: [Watch here](https://drive.google.com/file/d/1sBfZXJrlT83Rtihq3JFvIydy2k8ZJPuD/view?usp=drive_link)
- Live site: [GitHub Pages](https://kumarialeti.github.io/OSDHack/)

## Problem Statement
Participants need to demonstrate **on‑device AI** that runs **entirely in the browser** without any external APIs. The challenge is to provide useful AI‑powered functionality (retrieval‑augmented generation, resume evaluation) while keeping the solution lightweight, private, and offline.

## Solution
We built a single‑page web app that runs a **local LLM** (Qwen2‑0.5B‑Instruct) via **WebLLM**, uses **ONNX Runtime Web** to generate **sentence embeddings** (all‑MiniLM‑L6‑v2), stores everything in **IndexedDB**, and provides three core features:

1. **RAG** – drag‑and‑drop multiple PDFs/Markdown files, build a local vector index, and ask questions with citations.
2. **ATS Evaluation** – upload a resume and a job description, get a deterministic keyword‑match score and optional LLM‑generated improvement suggestions.
3. **Privacy Dashboard** – view local storage usage and clear all data with a single click.

All inference runs in the browser; no network calls are made after the initial model download, satisfying the on‑device requirement.

## Features
- **100% Offline AI**: Runs entirely in the browser using WebGPU/WASM, keeping all data on-device.
- **Multi-Document RAG**: Upload and search across multiple PDFs and Text documents simultaneously.
- **Smart ATS Evaluator**: Get precise keyword-matching scores and actionable AI resume-improvement advice.
- **Privacy Dashboard**: Complete transparency over your local data with a 1-click wipe option.
- **Zero Cloud APIs**: Uses local WebLLM for generation and Transformers.js for semantic embeddings.

## On‑Device AI Compliance
- **No external APIs** – all model files are cached locally via the browser’s Cache Storage.
- **Local storage** – documents, embeddings, and model weights stay in IndexedDB / Cache Storage.
- **Privacy‑first UI** – the dashboard shows exactly what is stored and offers a clear‑all button.
- **Web‑GPU acceleration** – WebLLM prefers `webgpu` for fast inference; falls back to `webgl`.

## Architecture Diagram
```mermaid
graph TD;
    A["User Interface"] -->|"Upload Files"| B["File Handlers (utils.js)"]
    B --> C["Embedding Engine (Transformers.js)"]
    C --> D["Vector Store (Dexie IndexedDB)"]
    D --> E["Similarity Search (cosine similarity)"]
    E --> F["Prompt Construction"]
    F --> G["WebLLM (Qwen2-0.5B-Instruct)"]
    G --> H["Answer Generation"]
    subgraph ATS
        I["Resume Upload"] --> J["Keyword Extraction"]
        K["Job Description Upload"] --> J
        J --> L["Deterministic Scoring"]
        L --> M["Optional LLM Advice"]
        M --> A
    end
    subgraph Privacy
        N["Dashboard UI"] --> O["IndexedDB Stats"]
        N --> P["Cache Storage Stats"]
        N --> Q["Clear All"]
    end
```

## AI Pipeline
The core functionality relies on a streamlined local AI pipeline:
1. **Document Upload**: PDF.js extracts raw text from uploaded files directly in the browser.
2. **Chunking**: Text is split into semantic chunks of 600 characters with 100 characters of overlap.
3. **Embedding**: `Transformers.js` uses the `all-MiniLM-L6-v2` model to convert chunks into dense vector embeddings.
4. **Storage**: Vectors are indexed locally using `Dexie.js` (IndexedDB).
5. **Retrieval**: When a query is made, it is embedded and matched against the local index using Cosine Similarity.
6. **Generation**: The top retrieved chunks are injected into a prompt and sent to `WebLLM` (Qwen2-0.5B-Instruct) to generate a conversational answer with citations.

## Tech Stack
- **HTML / CSS** – vanilla, glassmorphism design, responsive layout.
- **JavaScript (ES modules)** – modular code base.
- **WebLLM (MLC‑AI)** – runs Qwen2‑0.5B‑Instruct locally.
- **ONNX Runtime Web** – inference for sentence embeddings.
- **Dexie.js** – thin wrapper over IndexedDB for persisting vectors/documents.
- **PDF.js** – client‑side PDF parsing.
- **Marked.js** – markdown rendering (placeholder UI).

## Setup
1. **Clone / copy the project** into a folder, e.g. `osdhack2026_rag`.
2. **Place model files**:
   - `models/Qwen2-0.5B-Instruct/model.onnx` (WebLLM will download automatically on first run).
   - `models/all-MiniLM-L6-v2.onnx` – the embedding model used by `vectorStore.js`.
   - Ensure both files are reachable relative to `index.html` (the default paths work).
3. **Start a local server** (Python example):
   ```bash
   cd osdhack2026_rag
   python -m http.server 8000
   ```
4. Open <http://localhost:8000> in a modern browser (Chrome/Edge recommended for WebGPU support).

## Demo Steps
1. **Load Model** – click **Load Local AI Model**; progress bar shows download.
2. **RAG** – drag‑and‑drop one or more PDFs/Markdown files, click **Build Index**, then ask a question.
3. **ATS** – upload a resume file and a job description, click **Evaluate**, view the score and missing keywords; optional LLM advice appears if the model is loaded.
4. **Privacy** – open the **Privacy** tab to see storage usage and press **Clear All** to wipe data.

## Screenshots

*(Please place your screenshots in an `assets/` folder and name them as follows)*

1. **Home / Model Loading**  
   ![Home](assets/screenshot1.png)
2. **RAG Pipeline & Retrieval**  
   ![RAG](assets/screenshot2.png)
3. **ATS Resume Evaluator**  
   ![ATS](assets/screenshot3.png)
4. **Privacy Dashboard**  
   ![Privacy](assets/screenshot4.png)

## Future Improvements
- Add **vector quantization** to reduce memory footprint.
- Support **incremental indexing** for larger corpora.
- Integrate **speech‑to‑text** for voice queries.
- Provide **export / import** of the vector store.
- Explore **fine‑tuning** the LLM on domain‑specific data (still offline).

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---
*Built for OSDHack 2026 – a fully offline, privacy‑preserving AI demo.*
