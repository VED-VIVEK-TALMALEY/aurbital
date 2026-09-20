export type AreaType = "point" | "polygon";

export interface SelectedAreaPayload {
  coordinates: number[][];
  center: { lat: number; lng: number };
  areaType: AreaType;
  imageDataUrl?: string;
  imageCaptureError?: string;
}

export interface GeoAnalysis {
  summary: string;
  demographics: string;
  economy: string;
  environment: string;
  history: string;
  infrastructure: string;
  prediction: string;
}

export interface ChatMessage {
  role: "user" | "assistant";
  content: string;
  ts: number;
}

export interface ResearchFeedbackPayload {
  sessionId?: string;
  area: SelectedAreaPayload;
  question: string;
  modelAnswer: string;
  researcherScore: number;
  metrics?: {
    relevance?: number;
    factuality?: number;
    usefulness?: number;
  };
  notes?: string;
  label?: string;
}

export interface ManualTrainPayload {
  sessionId?: string;
  epochs: number;
  learningRate: number;
  batchSize: number;
  gradAccum: number;
  rewardWeight: number;
  targetMetric: "vqa_accuracy" | "f1" | "cider" | "custom";
  minTargetValue: number;
  datasetPath?: string;
  notes?: string;
}

export interface MultimodalResearchPayload {
  sessionId?: string;
  question: string;
  textContext?: string;
  area?: SelectedAreaPayload;
  imageDataUrls?: string[];
  videoFramesDataUrls?: string[];
}

export interface AuthUser {
  username: string;
  role: string;
}

export interface DashboardMetrics {
  training: {
    bestValLoss: number | null;
    epochs: number | null;
    globalSteps: number | null;
    lastEpoch: {
      epoch: number | null;
      trainLoss: number | null;
      valLoss: number | null;
      lr: number | null;
    } | null;
  };
  evaluation: {
    day5: Record<string, unknown> | null;
    baseline: Record<string, unknown> | null;
  };
  rl: {
    feedbackCount: number;
  };
}
