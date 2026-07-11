// ats.js – deterministic ATS scoring (keyword matching) and optional LLM advice

import { readFileAsync, parsePDF, markdownToText, showToast } from "./utils.js";
import { generateText } from "./model.js"; // optional LLM advice

/**
 * Simple keyword extraction from a job description.
 * Splits on non‑word characters, filters stop words, and returns unique lower‑cased terms.
 */
function extractKeywords(text) {
  const stopWords = new Set([
    'the','and','or','a','an','to','of','in','for','with','on','at','by','as','is','are','was','were','be','been','being','will','shall','should','could','would','may','might','must','can','do','does','did','has','have','had','having','its','their','our','your','my','its','that','this','these','those','but','if','then','else','when','while','about','above','below','into','over','under','again','once','here','there','where','why','how','all','any','both','each','few','more','most','other','some','such','no','nor','not','only','own','same','so','than','too','very','s','t','can','will','just','don','should','now'
  ]);
  return Array.from(new Set(
    text
      .toLowerCase()
      .split(/[^a-z0-9]+/)
      .filter(w => w && !stopWords.has(w) && w.length > 2)
  ));
}

/**
 * Deterministic ATS evaluation.
 * Returns a numeric score (0‑100) and arrays of matched / missing keywords.
 * @param {string} resumeText
 * @param {string} jdText
 * @returns {{score:number, matchedKeywords:string[], missingKeywords:string[]}}
 */
export function evaluateATS(resumeText, jdText) {
  const jdKeywords = extractKeywords(jdText);
  const resumeWords = new Set(
    resumeText
      .toLowerCase()
      .split(/[^a-z0-9]+/)
      .filter(Boolean)
  );
  const matched = jdKeywords.filter(k => resumeWords.has(k));
  const missing = jdKeywords.filter(k => !resumeWords.has(k));
  const score = Math.round((matched.length / jdKeywords.length) * 100);
  return { score, matchedKeywords: matched, missingKeywords: missing };
}

/**
 * Optional LLM advice generator – uses the already loaded WebLLM engine.
 * @param {object} llmEngine – engine returned from initWebLLM()
 * @param {string} resumeText
 * @param {string[]} missingKeywords
 * @returns {Promise<string>} advice text
 */
export async function generateATSAdvice(llmEngine, resumeText, missingKeywords) {
  if (!llmEngine) return '';
  const prompt = `You are an HR assistant. Based on the following resume text, suggest concrete improvements to cover the missing keywords: ${missingKeywords.join(', ')}. Keep the advice concise (2‑3 sentences). Resume:\n${resumeText}`;
  const advice = await generateText(llmEngine, prompt);
  return advice.trim();
}
