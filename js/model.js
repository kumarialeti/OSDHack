// model.js – WebLLM integration for Qwen2-0.5B-Instruct

// Load the WebLLM library from CDN (make sure the script is included in index.html if not using import)
// The library exposes a global `mlc` object.

export async function initWebLLM() {
  // Ensure the mlc namespace is available
  if (!window.mlc) {
    // Dynamically load the script if not present
    await loadScript('https://cdn.jsdelivr.net/npm/@mlc-ai/web-llm@0.2.7/dist/web-llm.js');
  }

  // Create the engine with the desired model
  const engine = await mlc.ChatEngine.create({
    model: 'Qwen2-0.5B-Instruct', // model name supported by WebLLM
    // model path can be a URL; the library will download and cache it.
    // We'll let it use the default CDN location.
    // You can customize the local cache size if needed.
    device: 'webgpu', // Prefer WebGPU for speed, fallback to webgl
  });
  // Warm up the model (optional)
  await engine.preload(1);
  return engine;
}

/**
 * Generate a response from the LLM.
 * @param {object} engine – the WebLLM engine returned from initWebLLM
 * @param {string} prompt – the full prompt including system instructions and context
 * @returns {Promise<string>} the generated text (trimmed)
 */
export async function generateText(engine, prompt) {
  const result = await engine.chat.completions.create({
    messages: [{ role: 'user', content: prompt }],
    temperature: 0.7,
    max_gen_len: 256,
  });
  return result.choices[0].message.content.trim();
}

/** Helper to load an external script dynamically */
function loadScript(src) {
  return new Promise((resolve, reject) => {
    const s = document.createElement('script');
    s.src = src;
    s.onload = resolve;
    s.onerror = reject;
    document.head.appendChild(s);
  });
}
