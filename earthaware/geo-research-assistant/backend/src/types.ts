export interface AreaPayload {
  coordinates: number[][];
  center: { lat: number; lng: number };
  areaType: "point" | "polygon";
  imageDataUrl?: string;
  imageCaptureError?: string;
  sessionId?: string;
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

export interface MultimodalResearchPayload {
  sessionId?: string;
  question: string;
  textContext?: string;
  area?: AreaPayload;
  imageDataUrls?: string[];
  videoFramesDataUrls?: string[];
}

export interface FeedbackPayload {
  sessionId?: string;
  area: AreaPayload;
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
