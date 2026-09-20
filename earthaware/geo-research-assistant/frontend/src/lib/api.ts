import type {
  AuthUser,
  DashboardMetrics,
  GeoAnalysis,
  ManualTrainPayload,
  MultimodalResearchPayload,
  ResearchFeedbackPayload,
  SelectedAreaPayload,
} from "../types";

const API_BASE = import.meta.env.VITE_API_BASE_URL ?? "http://localhost:8080";

export async function login(username: string, password: string) {
  const res = await fetch(`${API_BASE}/api/auth/login`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ username, password }),
  });
  if (!res.ok) {
    const text = await res.text();
    throw new Error(`Login failed: ${res.status} ${text}`);
  }
  return res.json() as Promise<{ token: string; user: AuthUser }>;
}

export async function getDashboardMetrics(token: string) {
  const res = await fetch(`${API_BASE}/api/research/dashboard-metrics`, {
    headers: { Authorization: `Bearer ${token}` },
  });
  if (!res.ok) {
    const text = await res.text();
    throw new Error(`Metrics fetch failed: ${res.status} ${text}`);
  }
  return res.json() as Promise<DashboardMetrics>;
}

export interface AreaResponse {
  analysis: GeoAnalysis;
  assistantMessage: string;
  sessionId: string;
}

export async function runMultimodalResearch(payload: MultimodalResearchPayload): Promise<AreaResponse> {
  const res = await fetch(`${API_BASE}/api/research/multimodal`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  if (!res.ok) {
    const text = await res.text();
    throw new Error(`Multimodal research failed: ${res.status} ${text}`);
  }
  return res.json();
}

export async function analyzeArea(payload: SelectedAreaPayload, sessionId?: string): Promise<AreaResponse> {
  const res = await fetch(`${API_BASE}/api/research/area`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ ...payload, sessionId }),
  });
  if (!res.ok) {
    const text = await res.text();
    throw new Error(`Area analysis failed: ${res.status} ${text}`);
  }
  return res.json();
}

export async function askFollowup(sessionId: string, question: string, area: SelectedAreaPayload) {
  const res = await fetch(`${API_BASE}/api/research/chat`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ sessionId, question, area }),
  });
  if (!res.ok) {
    const text = await res.text();
    throw new Error(`Chat failed: ${res.status} ${text}`);
  }
  return res.json() as Promise<{ reply: string }>;
}

export async function submitFeedback(payload: ResearchFeedbackPayload) {
  const res = await fetch(`${API_BASE}/api/research/feedback`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  if (!res.ok) {
    const text = await res.text();
    throw new Error(`Feedback submit failed: ${res.status} ${text}`);
  }
  return res.json() as Promise<{ success: boolean; reward: number; feedbackPath: string }>;
}

export async function startManualTraining(payload: ManualTrainPayload) {
  const res = await fetch(`${API_BASE}/api/research/manual-train`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  if (!res.ok) {
    const text = await res.text();
    throw new Error(`Manual training failed: ${res.status} ${text}`);
  }
  return res.json() as Promise<{ success: boolean; jobId: string; status: string; logPath: string; note: string }>;
}

export async function getManualTrainingStatus(jobId: string) {
  const res = await fetch(`${API_BASE}/api/research/manual-train/${jobId}`);
  if (!res.ok) {
    const text = await res.text();
    throw new Error(`Status fetch failed: ${res.status} ${text}`);
  }
  return res.json() as Promise<{
    jobId: string;
    status: "queued" | "running" | "completed" | "failed";
    startedAt: string;
    finishedAt?: string;
    logPath: string;
    exitCode?: number;
    error?: string;
  }>;
}
