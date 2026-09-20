// OpenAI import removed; using local model via Python script
import type { AreaPayload, GeoAnalysis, MultimodalResearchPayload } from "../types.js";
import { exec } from "child_process";
import { promisify } from "util";
import path from "path";
import { fileURLToPath } from "url";

const execPromise = promisify(exec);
const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const RUN_MODEL_SCRIPT = path.join(__dirname, "../../run_model.py");

type ChatTurn = { role: string; content: string };
type QueryIntent = "geospatial" | "methodology" | "model_training" | "coding" | "general";

interface LocalVisionMetadata {
  classification?: { label?: string; confidence?: number };
  spectral_indices?: { NDVI?: number; NDWI?: number; NDBI?: number };
  image_size?: number[];
}

interface LocalVisionResult {
  analysisText: string;
  metadata?: LocalVisionMetadata;
}

const provider = (process.env.AI_PROVIDER || (process.env.OPENAI_API_KEY ? "openai" : "local")).toLowerCase();
const model = process.env.OPENAI_MODEL || "gpt-4o";
const localApiBase = (process.env.LOCAL_MODEL_API_BASE || "http://localhost:8000").replace(/\/$/, "");
const localMaxTokens = Number(process.env.LOCAL_MODEL_MAX_TOKENS || 320);
const localTemperature = Number(process.env.LOCAL_MODEL_TEMPERATURE || 0.2);
const localApiTokenLimit = Number(process.env.LOCAL_MODEL_MAX_TOKENS_HARD_LIMIT || 512);

// OpenAI client removed; not needed for local model

const STOP_WORDS = new Set([
  "the", "a", "an", "is", "are", "was", "were", "be", "to", "of", "for", "in", "on", "at", "by", "with", "and",
  "or", "if", "then", "that", "this", "it", "as", "from", "about", "into", "how", "what", "why", "when", "where",
  "can", "could", "would", "should", "do", "does", "did", "i", "you", "we", "they", "he", "she", "them", "our",
]);

function buildPrompt(area: AreaPayload, followup?: string): string {
  const coords = JSON.stringify(area.coordinates);
  const followupText = followup ? `\nFollow-up question: ${followup}` : "";

  return `You are a geospatial research analyst.

Analyze the selected geographic region using the following coordinates:
${coords}
Center: lat=${area.center.lat}, lng=${area.center.lng}
Type: ${area.areaType}
${followupText}

Provide structured analysis including:
- Geographic Overview
- Demographics
- Economic Activity
- Environmental Factors
- Infrastructure
- Historical or Strategic Importance
- Prediction (what may happen next)

Respond in strict JSON with keys:
summary, demographics, economy, environment, history, infrastructure, prediction`;
}

function normalizeAnalysis(parsed: any): GeoAnalysis {
  return {
    summary: String(parsed?.summary || "No summary available."),
    demographics: String(parsed?.demographics || "No demographics available."),
    economy: String(parsed?.economy || "No economy insight available."),
    environment: String(parsed?.environment || "No environment insight available."),
    history: String(parsed?.history || "No history available."),
    infrastructure: String(parsed?.infrastructure || "No infrastructure insight available."),
    prediction: String(parsed?.prediction || "No prediction available."),
  };
}

function tryParseJsonObject(text: string): any {
  const raw = (text || "").trim();
  if (!raw) return {};

  try {
    return JSON.parse(raw);
  } catch {
    const start = raw.indexOf("{");
    const end = raw.lastIndexOf("}");
    if (start >= 0 && end > start) {
      const sliced = raw.slice(start, end + 1);
      try {
        return JSON.parse(sliced);
      } catch {
        return {};
      }
    }
    return {};
  }
}

function looksLikePromptEcho(text: string): boolean {
  const t = (text || "").toLowerCase();
  return t.includes("you are a geospatial research analyst") || t.includes("respond in strict json") || t.split("assistant:").length > 2;
}

function sanitizeVisionText(text: string): string {
  let t = (text || "").trim();
  if (!t) return "";

  
  t = t.replace(/\[[^\]]{1,24}\]/g, " ").replace(/[\r\n\t]+/g, " ");
  t = t.replace(/\s{2,}/g, " ").trim();

  
  t = t
    .replace(/analyze this selected satellite patch\.?/gi, "")
    .replace(/identify dominant landcover cues.*?be concrete\.?/gi, "")
    .trim();

  return t;
}

function looksCorruptedVisionText(text: string): boolean {
  const t = (text || "").trim();
  if (!t) return false;
  const lowered = t.toLowerCase();
  if (looksLikePromptEcho(lowered)) return true;
  if (/(eci-qa|assistant:|user:|system:)/i.test(lowered)) return true;

  const junkChars = (t.match(/[\[\]\|\\]/g) || []).length;
  if (junkChars >= 10) return true;

  const tokens = t.split(/\s+/);
  if (tokens.length >= 8) {
    const unique = new Set(tokens.map((x) => x.toLowerCase())).size;
    if (unique / tokens.length < 0.35) return true;
  }

  return false;
}

function isCudaRuntimeFault(message: string): boolean {
  const m = (message || "").toLowerCase();
  return m.includes("cuda") && (m.includes("device-side assert") || m.includes("launch blocking") || m.includes("cublas") || m.includes("kernel"));
}

function bboxFromCoordinates(coords: number[][]): { minLng: number; minLat: number; maxLng: number; maxLat: number } {
  let minLng = Infinity;
  let minLat = Infinity;
  let maxLng = -Infinity;
  let maxLat = -Infinity;
  for (const [lng, lat] of coords) {
    minLng = Math.min(minLng, lng);
    minLat = Math.min(minLat, lat);
    maxLng = Math.max(maxLng, lng);
    maxLat = Math.max(maxLat, lat);
  }
  return { minLng, minLat, maxLng, maxLat };
}

function estimateSpatialMetrics(area: AreaPayload) {
  if (area.areaType !== "polygon" || area.coordinates.length < 4) {
    const lng = area.center.lng;
    const lat = area.center.lat;
    return { areaKm2: 0, latKm: 0, lonKm: 0, minLng: lng, minLat: lat, maxLng: lng, maxLat: lat };
  }

  const { minLng, minLat, maxLng, maxLat } = bboxFromCoordinates(area.coordinates);
  const avgLatRad = ((minLat + maxLat) / 2) * (Math.PI / 180);
  const latKm = Math.max(0, (maxLat - minLat) * 111.32);
  const lonKm = Math.max(0, (maxLng - minLng) * 111.32 * Math.cos(avgLatRad));
  const areaKm2 = Math.max(0, latKm * lonKm);
  return { areaKm2, latKm, lonKm, minLng, minLat, maxLng, maxLat };
}

function compactCoordinates(area: AreaPayload): string {
  const coords = area.coordinates || [];
  if (!coords.length) return "[]";
  if (coords.length <= 6) return JSON.stringify(coords);
  const head = coords.slice(0, 3);
  const tail = coords.slice(-2);
  return `${JSON.stringify(head)} ... ${JSON.stringify(tail)} (n=${coords.length})`;
}

function inferPredictionFromText(text: string): string {
  const t = text.toLowerCase();
  if (/(water|flood|river|reservoir|wet|ndwi)/.test(t)) {
    return "Near-term risk: seasonal water spread/flooding in low-lying pockets; monitor rainfall-triggered expansion after storm events.";
  }
  if (/(urban|built|road|construction|settlement|ndbi)/.test(t)) {
    return "Trend signal: likely built-up expansion and surface sealing pressure; monitor impervious growth and heat-stress proxies over 6-12 months.";
  }
  if (/(forest|vegetation|crop|agri|farmland|ndvi)/.test(t)) {
    return "Trend signal: vegetation/crop condition likely seasonal; monitor NDVI anomaly and moisture stress for upcoming crop cycle.";
  }
  return "Prediction requires multi-temporal stack; run monthly change detection before making operational decisions.";
}

function extractLatestIndicesFromHistory(history: ChatTurn[]): { ndvi?: number; ndwi?: number; ndbi?: number } {
  const assistants = [...history].reverse().filter((h) => h.role === "assistant");
  for (const msg of assistants) {
    const t = msg.content || "";
    const ndvi = t.match(/NDVI\s*=\s*(-?\d+(\.\d+)?)/i);
    const ndwi = t.match(/NDWI\s*=\s*(-?\d+(\.\d+)?)/i);
    const ndbi = t.match(/NDBI\s*=\s*(-?\d+(\.\d+)?)/i);
    if (ndvi || ndwi || ndbi) {
      return {
        ndvi: ndvi ? Number(ndvi[1]) : undefined,
        ndwi: ndwi ? Number(ndwi[1]) : undefined,
        ndbi: ndbi ? Number(ndbi[1]) : undefined,
      };
    }
  }
  return {};
}

function buildFloodDroughtAnswer(area: AreaPayload, history: ChatTurn[], question: string): string {
  const m = estimateSpatialMetrics(area);
  const idx = extractLatestIndicesFromHistory(history);
  const ndvi = idx.ndvi;
  const ndwi = idx.ndwi;
  const ndbi = idx.ndbi;

  let floodRisk = "moderate";
  let droughtRisk = "moderate";
  let rationale = "Risk estimate is based on recent spectral indicators from this session.";

  if (typeof ndwi === "number" && typeof ndvi === "number") {
    if (ndwi >= 0.12) floodRisk = "elevated";
    if (ndwi <= -0.05) floodRisk = "low";
    if (ndvi <= 0.10 && ndwi <= 0.02) droughtRisk = "elevated";
    if (ndvi >= 0.35 && ndwi >= 0.05) droughtRisk = "low";
    rationale = `Observed NDVI=${ndvi.toFixed(3)}, NDWI=${ndwi.toFixed(3)}${typeof ndbi === "number" ? `, NDBI=${ndbi.toFixed(3)}` : ""}.`;
  } else if (typeof ndwi === "number") {
    floodRisk = ndwi >= 0.12 ? "elevated" : ndwi <= -0.05 ? "low" : "moderate";
    rationale = `Observed NDWI=${ndwi.toFixed(3)} from selected patch.`;
  } else {
    rationale = "No reliable NDWI/NDVI found in session history; this is a conservative estimate.";
  }

  const q = question.toLowerCase();
  const wantsBinary = q.includes("flood or drought");

  if (wantsBinary) {
    const dominant = floodRisk === "elevated" && droughtRisk !== "elevated"
      ? "flood risk is currently more likely than drought."
      : droughtRisk === "elevated" && floodRisk !== "elevated"
      ? "drought/moisture-stress risk is currently more likely than flooding."
      : "both risks are currently moderate/uncertain; temporal stack is required for high confidence.";
    return [
      `Flood-Drought Assessment for selected area (~${m.areaKm2.toFixed(2)} km^2):`,
      dominant,
      `Flood risk: ${floodRisk}. Drought risk: ${droughtRisk}.`,
      rationale,
      "For operational decision: compare with last-year same-season image and 30-day rainfall anomaly.",
    ].join("\n");
  }

  return [
    `Flood Risk Assessment for selected area (~${m.areaKm2.toFixed(2)} km^2):`,
    `Flood risk: ${floodRisk}. Drought risk: ${droughtRisk}.`,
    rationale,
    "Recommendation: upload previous-year same-season image in Multimodal mode for stronger change confirmation.",
  ].join("\n");
}

function imageGroundedAnalysis(area: AreaPayload, vision: LocalVisionResult): GeoAnalysis {
  const m = estimateSpatialMetrics(area);
  const label = vision.metadata?.classification?.label ?? "unknown_landcover";
  const conf = typeof vision.metadata?.classification?.confidence === "number" ? vision.metadata?.classification?.confidence : null;

  const ndvi = vision.metadata?.spectral_indices?.NDVI;
  const ndwi = vision.metadata?.spectral_indices?.NDWI;
  const ndbi = vision.metadata?.spectral_indices?.NDBI;

  const indexLine = [
    typeof ndvi === "number" ? `NDVI=${ndvi.toFixed(3)}` : "NDVI=n/a",
    typeof ndwi === "number" ? `NDWI=${ndwi.toFixed(3)}` : "NDWI=n/a",
    typeof ndbi === "number" ? `NDBI=${ndbi.toFixed(3)}` : "NDBI=n/a",
  ].join(", ");

  const classLine = conf != null ? `${label} (confidence ${conf.toFixed(3)})` : label;
  const rawObs = sanitizeVisionText(vision.analysisText || "");
  const modelObs = looksCorruptedVisionText(rawObs) ? "" : rawObs;

  const envInterpretation = (() => {
    if (typeof ndwi === "number" && ndwi > 0.15) return "Water/moisture signal is elevated in the selected patch.";
    if (typeof ndvi === "number" && ndvi > 0.35) return "Vegetation signal is relatively strong in the selected patch.";
    if (typeof ndbi === "number" && ndbi > 0.20) return "Built-up/impervious signal is elevated in the selected patch.";
    return "Mixed or weak spectral dominance is observed; require temporal stack for confident trend attribution.";
  })();

  const prediction = inferPredictionFromText(`${label} ${indexLine} ${modelObs}`);

  return {
    summary: `Image-grounded finding: class=${classLine}. Spatial extent ~${m.areaKm2.toFixed(2)} km^2 (${m.latKm.toFixed(2)} km x ${m.lonKm.toFixed(2)} km). Spectral indicators: ${indexLine}.`,
    demographics: "No direct demographic signal from imagery alone. For research-grade demographic inference, intersect this polygon with census + settlement layers.",
    economy: `Land-surface evidence suggests ${label.replace(/_/g, " ")}. Economic interpretation should link this class with district-level land-use and market-access datasets for quantification.`,
    environment: `${envInterpretation}${modelObs ? ` Model observation: ${modelObs}` : ""}`,
    history: "For historical significance, compare this patch against archived imagery (monthly/seasonal) and administrative event records.",
    infrastructure: "Extract roads/buildings/utilities within this box (OSM/local GIS) and compute density/connectivity to quantify infrastructure pressure.",
    prediction,
  };
}

function deterministicGeoAnalysis(area: AreaPayload): GeoAnalysis {
  const m = estimateSpatialMetrics(area);
  return {
    summary: `Geometry-only fallback: center (${area.center.lat.toFixed(4)}, ${area.center.lng.toFixed(4)}), area ~${m.areaKm2.toFixed(2)} km^2. Image patch was unavailable for content analysis.`,
    demographics: "Demographic inference requires overlay with census and settlement datasets.",
    economy: "Economic inference requires land-use/economic proxy overlays (night lights, road access, market distance).",
    environment: "Image content was not available in this request; run boxed image analysis for concrete environmental findings.",
    history: "Use multi-date imagery and archived records for historical attribution.",
    infrastructure: "Use vector overlays to quantify roads/buildings/utilities within the selected geometry.",
    prediction: "Prediction requires image-grounded features and temporal stack; rerun with valid boxed image capture.",
  };
}

function mergeWithNarrativeFallback(area: AreaPayload, parsed: any, rawText: string): GeoAnalysis {
  const structured = normalizeAnalysis(parsed);
  const hasAnyStructured = [structured.summary, structured.demographics, structured.economy, structured.environment, structured.history, structured.infrastructure, structured.prediction].some(
    (v) => !v.toLowerCase().startsWith("no ")
  );

  const cleanText = (rawText || "").trim();

  if (hasAnyStructured && !looksLikePromptEcho(structured.summary)) {
    if (structured.prediction.toLowerCase().startsWith("no ")) {
      structured.prediction = inferPredictionFromText(structured.summary);
    }
    return structured;
  }

  if (cleanText && !looksLikePromptEcho(cleanText)) {
    return {
      summary: cleanText,
      demographics: "Demographic interpretation requires external census overlay.",
      economy: "Economic interpretation should be validated with proxy datasets.",
      environment: cleanText,
      history: "Historical attribution requires multi-date verification.",
      infrastructure: "Infrastructure quantification requires vector overlays.",
      prediction: inferPredictionFromText(cleanText),
    };
  }

  return deterministicGeoAnalysis(area);
}

function normalizeQuestion(question: string): string {
  const q = question.replace(/\s+/g, " ").trim();
  if (!q) return "";
  return /[.?!]$/.test(q) ? q : `${q}?`;
}

function detectIntent(question: string): QueryIntent {
  const q = question.toLowerCase();
  if (/(ndvi|ndwi|ndbi|land ?cover|soil|crop|flood|deforestation|terrain|satellite|geo|raster|vector|bbox|polygon)/.test(q)) return "geospatial";
  if (/(method|evaluation|metric|baseline|ablation|experiment|hypothesis|paper|research)/.test(q)) return "methodology";
  if (/(train|fine ?tune|lora|qlora|epoch|learning rate|loss|reward|rl|reinforcement)/.test(q)) return "model_training";
  if (/(code|api|endpoint|typescript|python|bug|error|stack trace|compile|build script|build error|build failed)/.test(q)) return "coding";
  return "general";
}

function extractKeywords(question: string): string[] {
  const tokens = question
    .toLowerCase()
    .replace(/[^a-z0-9\s]/g, " ")
    .split(/\s+/)
    .filter((t) => t.length > 2 && !STOP_WORDS.has(t));

  const freq = new Map<string, number>();
  for (const t of tokens) freq.set(t, (freq.get(t) || 0) + 1);
  return [...freq.entries()].sort((a, b) => b[1] - a[1]).slice(0, 8).map(([w]) => w);
}

function isLowQualityAnswer(answer: string, normalizedQuestion: string): boolean {
  const a = (answer || "").trim();
  if (!a || a.length < 32) return true;
  if (looksLikePromptEcho(a)) return true;
  if (/(instructions:|user question:|detected intent:|keywords:|conversation context:|your previous answer was low quality)/i.test(a)) return true;
  if (/(^|\n)\s*(user|assistant|system)\s*:/i.test(a)) return true;
  if (/you are earthaware/i.test(a)) return true;
  if (/^\(\s*-?\d+(\.\d+)?\s*,\s*-?\d+(\.\d+)?\s*\)\s*,?\s*type\s*=\s*(point|polygon)/i.test(a)) return true;
  if (/extent_km/i.test(a) && !/[.!?]\s*$/.test(a)) return true;
  if (/^geospatial overview\s*$/i.test(a)) return true;
  const coordLike = a.match(/\[\s*-?\d+(\.\d+)?\s*(,\s*-?\d+(\.\d+)?\s*)*\]/g);
  if (coordLike && coordLike.length >= 4) return true;
  const numberRatio = (a.match(/[0-9.\-,\[\]]/g) || []).length / Math.max(a.length, 1);
  if (numberRatio > 0.65) return true;
  const q = normalizedQuestion.toLowerCase();
  const al = a.toLowerCase();
  if (q && al.startsWith(q)) return true;
  if (al.includes("i am unable") || al.includes("cannot answer") || al.includes("no response generated")) return true;
  return false;
}

function cleanAssistantAnswer(answer: string): string {
  let a = (answer || "").trim();
  if (!a) return "";

  
  a = a
    .replace(/<think>[\s\S]*?<\/think>/gi, " ")
    .replace(/```(?:thinking|reasoning)[\s\S]*?```/gi, " ")
    .replace(/^\s*(thoughts?|reasoning|analysis)\s*:\s*[\s\S]*?(?=^\s*(final answer|answer)\s*:|\Z)/gim, " ")
    .trim();

  
  const finalMarker = a.match(/(?:^|\n)\s*(final answer|answer)\s*:\s*/i);
  if (finalMarker && finalMarker.index != null) {
    a = a.slice(finalMarker.index + finalMarker[0].length).trim();
  }

  
  const lower = a.toLowerCase();
  const marker = lower.lastIndexOf("assistant:");
  if (marker >= 0) {
    a = a.slice(marker + "assistant:".length).trim();
  }

  
  a = a
    .replace(/^\s*instructions:\s*[\s\S]*?user question:\s*/i, "")
    .replace(/^\s*nlp preprocessed question:\s*.*$/gim, "")
    .replace(/^\s*detected intent:\s*.*$/gim, "")
    .replace(/^\s*keywords:\s*.*$/gim, "")
    .replace(/^\s*(user|assistant|system)\s*:\s*/gim, "")
    .replace(/^\s*return only final answer text.*$/gim, "")
    .replace(/^\s*do not include role labels.*$/gim, "")
    .replace(/^\s*area center:\s*.*$/gim, "")
    .replace(/^\s*prior context:\s*.*$/gim, "")
    .replace(/^\s*selected area coordinates.*$/gim, "")
    .replace(/^\s*estimated extent:\s*.*$/gim, "")
    .replace(/^\s*you are earthaware.*$/gim, "")
    .replace(/^\s*(let('|’)s think step by step|thinking aloud)\s*[:.-]?\s*$/gim, "")
    .trim();

  
  if (a.split("###").length > 6) {
    const firstUseful = a.split(/\n{2,}/).find((blk) => blk && !/(instructions:|user question:|detected intent:)/i.test(blk));
    if (firstUseful) a = firstUseful.trim();
  }

  
  const deduped: string[] = [];
  for (const line of a.split("\n")) {
    const t = line.trim();
    if (!t) continue;
    const last = deduped[deduped.length - 1] || "";
    if (last.toLowerCase() === t.toLowerCase()) continue;
    deduped.push(t);
  }
  a = deduped.join("\n").trim();

  return a;
}

function isPromptFragmentAnswer(answer: string): boolean {
  const a = (answer || "").toLowerCase();
  return /(return only final answer text|do not include role labels|prior context:|selected area coordinates|user question:|area center:|you are earthaware|type=polygon|type=point|extent_km)/i.test(a);
}

function isGreeting(question: string): boolean {
  const q = question.trim().toLowerCase();
  return /^(hi|hello|hey|hii|yo|namaste|good morning|good evening)[!. ]*$/.test(q);
}

function buildWeatherAnswer(area: AreaPayload, history: ChatTurn[], question: string): string {
  const m = estimateSpatialMetrics(area);
  const idx = extractLatestIndicesFromHistory(history);
  const ndvi = typeof idx.ndvi === "number" ? idx.ndvi.toFixed(3) : "n/a";
  const ndwi = typeof idx.ndwi === "number" ? idx.ndwi.toFixed(3) : "n/a";
  const ndbi = typeof idx.ndbi === "number" ? idx.ndbi.toFixed(3) : "n/a";

  const q = question.toLowerCase();
  const asksRain = /(rain|precipitation|monsoon|will rain|forecast)/i.test(q);

  if (asksRain) {
    return [
      `Weather Forecast Answer for selected area (~${m.areaKm2.toFixed(2)} km^2):`,
      "Rainfall cannot be predicted reliably from a single satellite snapshot alone.",
      `Current image signals: NDVI=${ndvi}, NDWI=${ndwi}, NDBI=${ndbi}.`,
      "Interpretation: these indicate current surface condition, not future rainfall.",
      "To answer 'will it rain here', integrate IMD/NOAA hourly forecast + last 7-day rainfall + cloud-motion sequence.",
    ].join("\n");
  }

  return [
    `Weather-Hydrology note for selected area (~${m.areaKm2.toFixed(2)} km^2):`,
    `Current indices NDVI=${ndvi}, NDWI=${ndwi}, NDBI=${ndbi}.`,
    "For future weather outcomes, use forecast datasets; image-only evidence is insufficient.",
  ].join("\n");
}

function fallbackFollowupAnswer(area: AreaPayload, question: string, history: ChatTurn[], reason: string): string {
  const intent = detectIntent(question);
  const m = estimateSpatialMetrics(area);
  const idx = extractLatestIndicesFromHistory(history);
  const ndvi = idx.ndvi != null ? idx.ndvi.toFixed(3) : "n/a";
  const ndwi = idx.ndwi != null ? idx.ndwi.toFixed(3) : "n/a";
  const ndbi = idx.ndbi != null ? idx.ndbi.toFixed(3) : "n/a";

  let core = "Use the current area findings as baseline and validate with temporal data.";
  const q = question.toLowerCase();
  if (q.includes("vegetation")) {
    core = "Vegetation change should be judged with NDVI trend and seasonal baseline. If NDVI drops across successive windows, stress/degradation risk increases.";
  } else if (q.includes("flood") || q.includes("water")) {
    core = "Flood/water risk should be assessed with NDWI rise and low-lying topography; confirm with rainfall nowcasts and drainage context.";
  } else if (q.includes("urban") || q.includes("built")) {
    core = "Urban expansion risk should be tracked using NDBI growth and road-network adjacency over time.";
  } else if (q.includes("building") || q.includes("construction") || q.includes("can buildings be made")) {
    core = "Construction suitability cannot be concluded from one image alone. Use slope/elevation, flood hazard, soil bearing capacity, and zoning constraints before deciding.";
  } else if (q.includes("previous year") || q.includes("last year") || q.includes("compare")) {
    core = "To compare with previous year, upload the earlier image in Multimodal input and run a change query. Current session has one-time slice, so temporal change cannot be measured yet.";
  }

  return [
    `Area context: ~${m.areaKm2.toFixed(2)} km^2 around (${area.center.lat.toFixed(4)}, ${area.center.lng.toFixed(4)}).`,
    `Current indicators: NDVI=${ndvi}, NDWI=${ndwi}, NDBI=${ndbi}.`,
    `Assessment: ${core}`,
    "Note: this response is conservative because model generation quality was unstable in this turn.",
  ].join("\n");
}

function dataUrlToBlob(dataUrl: string): { blob: Blob; ext: string } {
  const match = dataUrl.match(/^data:(image\/(png|jpeg|jpg));base64,(.+)$/i);
  if (!match) throw new Error("Invalid imageDataUrl format");
  const mime = match[1].toLowerCase();
  const ext = mime.includes("png") ? "png" : "jpg";
  const b64 = match[3];
  const buffer = Buffer.from(b64, "base64");
  const blob = new Blob([buffer], { type: mime.includes("jpg") ? "image/jpeg" : mime });
  return { blob, ext };
}

async function callLocalVisionAnalyze(area: AreaPayload, question: string): Promise<LocalVisionResult> {
  if (!area.imageDataUrl) throw new Error("imageDataUrl missing for local vision analysis");
  const { blob, ext } = dataUrlToBlob(area.imageDataUrl);
  const form = new FormData();
  form.append("image", blob, `selection.${ext}`);
  form.append("question", question);
  form.append("max_length", "300");
  form.append("temperature", String(localTemperature));

  const resp = await fetch(`${localApiBase}/analyze`, {
    method: "POST",
    body: form,
  });

  if (!resp.ok) {
    const txt = await resp.text();
    throw new Error(`Local vision analyze error (${resp.status}): ${txt}`);
  }

  const data = (await resp.json()) as { analysis?: string; metadata?: LocalVisionMetadata; success?: boolean };
  const cleanAnalysis = sanitizeVisionText(data.analysis || "");
  return { analysisText: looksCorruptedVisionText(cleanAnalysis) ? "" : cleanAnalysis, metadata: data.metadata };
}

async function callLocalChat(prompt: string): Promise<string> {
  return callLocalChatWithOptions(prompt);
}

async function callLocalChatWithOptions(prompt: string, opts?: { maxTokens?: number; temperature?: number }): Promise<string> {
  const maxTokens = Math.max(16, Math.min(localApiTokenLimit, opts?.maxTokens ?? localMaxTokens));
  const temperature = opts?.temperature ?? localTemperature;
  const send = async (mt: number) =>
    fetch(`${localApiBase}/chat`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ message: prompt, max_tokens: mt, temperature }),
    });

  let resp = await send(maxTokens);
  if (!resp.ok && resp.status === 422 && maxTokens > 512) {
    resp = await send(512);
  }

  if (!resp.ok) {
    const txt = await resp.text();
    throw new Error(`Local model API error (${resp.status}): ${txt}`);
  }

  const data = (await resp.json()) as { response?: string; answer?: string };
  return data.response || data.answer || "";
}

async function generateText(prompt: string): Promise<string> {
  if (provider === "local") {
    return callLocalChat(prompt);
  }
  throw new Error("AI provider is not set to local and no OpenAI client is configured.");
}

async function runPythonModel(payload: any): Promise<any> {
  const payloadStr = JSON.stringify(payload).replace(/"/g, '\\"');
  const command = `python "${RUN_MODEL_SCRIPT}" --payload "${payloadStr}"`;
  try {
    const { stdout } = await execPromise(command);
    return JSON.parse(stdout.trim());
  } catch (err: any) {
    console.error("Python script error:", err.message);
    throw new Error("Local model inference failed");
  }
}

function buildFollowupPrompt(area: AreaPayload, history: ChatTurn[], question: string): string {
  const normalizedQuestion = normalizeQuestion(question);
  const intent = detectIntent(normalizedQuestion);
  const keywords = extractKeywords(normalizedQuestion).join(", ") || "none";
  const shortHistory = history.slice(-6).map((m) => `${m.role}: ${m.content.slice(0, 400)}`).join("\n");

  const m = estimateSpatialMetrics(area);
  const coordsShort = compactCoordinates(area);
  return `You are EarthAware NLP Research Assistant.

NLP preprocessed question: ${normalizedQuestion}
Detected intent: ${intent}
Keywords: ${keywords}

Selected area coordinates (compact): ${coordsShort}
Area center: lat=${area.center.lat}, lng=${area.center.lng}
Area type: ${area.areaType}
Estimated extent: area_km2=${m.areaKm2.toFixed(3)}, ns_km=${m.latKm.toFixed(3)}, ew_km=${m.lonKm.toFixed(3)}

Conversation context:
${shortHistory}

Instructions:
1) Answer directly with concrete findings.
2) Include uncertainty only when needed.
3) Do not echo prompt text.
4) Never output raw coordinate arrays unless explicitly asked.

User question: ${normalizedQuestion}`;
}

function latestAssistantContext(history: ChatTurn[]): string {
  const latest = [...history].reverse().find((h) => h.role === "assistant")?.content || "";
  const compact = latest
    .replace(/\s+/g, " ")
    .replace(/`/g, "")
    .slice(0, 1400);
  return compact || "No previous assistant context available.";
}

function isReportRequest(question: string): boolean {
  return /(detailed|expert|full report|comprehensive|technical report|research report)/i.test(question);
}

function buildLocalLLMFollowupPrompt(area: AreaPayload, history: ChatTurn[], question: string): string {
  const m = estimateSpatialMetrics(area);
  const ctx = latestAssistantContext(history);
  return [
    "You are EarthAware geospatial LLM assistant.",
    "Return only final answer text for the user. Do not include role labels, prompt text, or internal instructions.",
    `Area center: (${area.center.lat.toFixed(5)}, ${area.center.lng.toFixed(5)}), type=${area.areaType}, extent_km2=${m.areaKm2.toFixed(3)}.`,
    `Prior context: ${ctx}`,
    `User question: ${question}`,
    "If asked for report, provide sections: Executive Summary, Evidence, Risk Assessment, Limitations, Next Data Needed.",
  ].join("\n");
}

function buildTemplateReport(area: AreaPayload, history: ChatTurn[], question: string): string {
  const m = estimateSpatialMetrics(area);
  const idx = extractLatestIndicesFromHistory(history);
  const ndvi = idx.ndvi != null ? idx.ndvi.toFixed(3) : "n/a";
  const ndwi = idx.ndwi != null ? idx.ndwi.toFixed(3) : "n/a";
  const ndbi = idx.ndbi != null ? idx.ndbi.toFixed(3) : "n/a";
  return [
    "Executive Summary",
    `Selected area (~${m.areaKm2.toFixed(2)} km^2) shows mixed land-surface signal with no extreme spectral dominance.`,
    "Evidence",
    `Observed indicators: NDVI=${ndvi}, NDWI=${ndwi}, NDBI=${ndbi}.`,
    "Risk Assessment",
    "Near-term hydrology and vegetation stress risks are moderate and need temporal confirmation (multi-date stack).",
    "Limitations",
    "Single-scene analysis cannot reliably forecast rainfall or event timing.",
    "Next Data Needed",
    "Add previous-year same-season imagery, 30-day rainfall anomaly, and local elevation/drainage layers.",
    `Requested task: ${question}`,
  ].join("\n");
}

export async function analyzeAreaWithLLM(area: AreaPayload): Promise<GeoAnalysis> {
  try {
    return await runPythonModel(area);
  } catch (err) {
    const msg = err instanceof Error ? err.message : String(err);
    if (isCudaRuntimeFault(msg)) {
      const safe = deterministicGeoAnalysis(area);
      safe.summary = `${safe.summary} (LLM vision path failed due GPU runtime fault; returned safe fallback.)`;
      safe.prediction = "GPU runtime fault detected. Restart model API, then rerun box analysis for higher-confidence prediction.";
      return safe;
    }
    throw err;
  }
}

export async function followupWithLLM(area: AreaPayload, history: ChatTurn[], question: string): Promise<string> {
  try {
    const payload = { type: "followup", area, history, question };
    const result = await runPythonModel(payload);
    return result.answer || result.summary || "Followup analysis complete.";
  } catch (err) {
    console.error(err);
    return "Follow-up failed due to model runtime issue. Please retry.";
  }
}

export async function multimodalResearchWithLLM(payload: MultimodalResearchPayload): Promise<GeoAnalysis> {
  try {
    const pyPayload = { type: "multimodal", ...payload };
    return await runPythonModel(pyPayload);
  } catch (err) {
    console.error(err);
    throw err;
  }
}
