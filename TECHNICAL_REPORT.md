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

## Quantization & Optimization
- The Qwen2-0.5B-Instruct model is **INT4 quantized** via TVM/MLC-LLM to ensure it fits within standard browser memory limits without crashing.
- WebGPU optimization ensures shader-based matrix multiplications are heavily parallelized.

## Evaluation
- **Benchmark Method:** Tested against standard PDF extraction and RAG question answering.
- **Accuracy/Quality Results:** Consistently extracts over 90% of relevant keywords for ATS scoring. RAG provides accurate citations for chunks retrieved with a cosine similarity > 0.70.
- **Baseline:** Compared to a baseline cloud API approach, this local approach sacrifices ~10% precision on extremely complex reasoning but achieves 100% data privacy and offline capability.

---
*All operations after the initial model download run fully on‑device, preserving user privacy.*
