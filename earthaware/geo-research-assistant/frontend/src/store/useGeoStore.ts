import { create } from "zustand";
import type { ChatMessage, GeoAnalysis, SelectedAreaPayload } from "../types";

interface GeoState {
  selectedArea: SelectedAreaPayload | null;
  sessionId: string | null;
  messages: ChatMessage[];
  analysis: GeoAnalysis | null;
  loading: boolean;
  cache: Record<string, GeoAnalysis>;
  setSelectedArea: (area: SelectedAreaPayload | null) => void;
  setSessionId: (id: string | null) => void;
  addMessage: (msg: ChatMessage) => void;
  setAnalysis: (a: GeoAnalysis | null) => void;
  setLoading: (v: boolean) => void;
  setCached: (key: string, a: GeoAnalysis) => void;
}

export const areaKey = (area: SelectedAreaPayload): string =>
  `${area.areaType}:${JSON.stringify(area.coordinates)}`;

export const useGeoStore = create<GeoState>((set) => ({
  selectedArea: null,
  sessionId: null,
  messages: [],
  analysis: null,
  loading: false,
  cache: {},
  setSelectedArea: (area) => set({ selectedArea: area }),
  setSessionId: (id) => set({ sessionId: id }),
  addMessage: (msg) => set((s) => ({ messages: [...s.messages, msg] })),
  setAnalysis: (analysis) => set({ analysis }),
  setLoading: (loading) => set({ loading }),
  setCached: (key, analysis) => set((s) => ({ cache: { ...s.cache, [key]: analysis } })),
}));
