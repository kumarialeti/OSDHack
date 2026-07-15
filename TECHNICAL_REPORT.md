# Technical Report

## Model
- **Qwen2-0.5B-Instruct** – 0.5 B parameters, pretrained for instruction following.

## Runtime
- **WebLLM** (MLC‑AI) runs the model entirely in the browser using WebGPU/WebGL.
- **ONNX Runtime Web** handles the embedding model inference.

## Embedding
- **Model:** `all-MiniLM-L6-v2` (384‑dimensional vectors).
- **Inference:** ONNX Runtime Web, executed locally.

## Vector Store
- **IndexedDB** (via Dexie.js) stores vectors and document metadata persistently.

## PDF Parsing
- **PDF.js** extracts raw text client‑side; no server involved.

## Deployment
- Hosted on **GitHub Pages** (`https://kumarialeti.github.io/OSDHack/`).
- Static assets; the only network request is the initial model download (cached thereafter).

## Performance
- **Average Answer Time:** 2–4 seconds per query.
- **Device Tested:** Windows laptop, Intel Core i5, 8 GB RAM.
- **Browser:** Chrome 118 with WebGPU enabled.

## Resource Usage
- **CPU:** Minimal after model load (WebGPU does most work).
- **GPU:** Utilizes WebGPU for both LLM and embedding inference.
- **Memory:** Peak ~350 MB (model + embeddings) during inference.

---
*All operations after the initial model download run fully on‑device, preserving user privacy.*
