import { useEffect, useMemo, useRef, useState } from "react";
import ReactMarkdown from "react-markdown";
import { analyzeArea, askFollowup, getManualTrainingStatus, runMultimodalResearch, startManualTraining, submitFeedback } from "../lib/api";
import { areaKey, useGeoStore } from "../store/useGeoStore";

export default function ChatPanel() {
  const { selectedArea, sessionId, setSessionId, messages, addMessage, setAnalysis, analysis, cache, setCached } = useGeoStore();

  const [input, setInput] = useState("");
  const [isAreaAnalyzing, setIsAreaAnalyzing] = useState(false);
  const [isChatLoading, setIsChatLoading] = useState(false);
  const [isMultimodalLoading, setIsMultimodalLoading] = useState(false);
  const [multimodalQuestion, setMultimodalQuestion] = useState("");
  const [multimodalContext, setMultimodalContext] = useState("");
  const [multimodalImages, setMultimodalImages] = useState<string[]>([]);
  const [multimodalFrames, setMultimodalFrames] = useState<string[]>([]);
  const [multimodalFilesLabel, setMultimodalFilesLabel] = useState("");

  const [researcherScore, setResearcherScore] = useState(0.5);
  const [relevance, setRelevance] = useState(0.5);
  const [factuality, setFactuality] = useState(0.5);
  const [usefulness, setUsefulness] = useState(0.5);
  const [feedbackNotes, setFeedbackNotes] = useState("");
  const [actionLoading, setActionLoading] = useState(false);

  const [epochs, setEpochs] = useState(2);
  const [learningRate, setLearningRate] = useState(0.00001);
  const [batchSize, setBatchSize] = useState(1);
  const [gradAccum, setGradAccum] = useState(8);
  const [rewardWeight, setRewardWeight] = useState(1);
  const [targetMetric, setTargetMetric] = useState<"vqa_accuracy" | "f1" | "cider" | "custom">("vqa_accuracy");
  const [minTargetValue, setMinTargetValue] = useState(0.7);
  const [jobId, setJobId] = useState<string | null>(null);

  const selectedAreaKey = useMemo(() => (selectedArea ? areaKey(selectedArea) : null), [selectedArea]);
  const inFlightAreaKeyRef = useRef<string | null>(null);
  const analyzedAreaKeyRef = useRef<string | null>(null);

  const canAsk = useMemo(() => Boolean(sessionId), [sessionId]);
  const fallbackArea = useMemo(
    () => ({
      areaType: "point" as const,
      center: { lat: 0, lng: 0 },
      coordinates: [[0, 0]] as number[][],
    }),
    []
  );

  useEffect(() => {
    if (!selectedArea || !selectedAreaKey) {
      analyzedAreaKeyRef.current = null;
      inFlightAreaKeyRef.current = null;
      return;
    }

    const cached = cache[selectedAreaKey];
    if (cached) {
      setAnalysis(cached);
      analyzedAreaKeyRef.current = selectedAreaKey;
      setIsAreaAnalyzing(false);
      return;
    }

    if (analyzedAreaKeyRef.current === selectedAreaKey || inFlightAreaKeyRef.current === selectedAreaKey) {
      return;
    }

    let active = true;
    inFlightAreaKeyRef.current = selectedAreaKey;
    setIsAreaAnalyzing(true);

    analyzeArea(selectedArea, sessionId ?? undefined)
      .then((res) => {
        if (!active) return;
        setSessionId(res.sessionId);
        setAnalysis(res.analysis);
        setCached(selectedAreaKey, res.analysis);
        analyzedAreaKeyRef.current = selectedAreaKey;
        addMessage({ role: "assistant", content: res.assistantMessage, ts: Date.now() });
      })
      .catch((err) => {
        if (!active) return;
        analyzedAreaKeyRef.current = null;
        const msg = err instanceof Error ? err.message : String(err);
        addMessage({ role: "assistant", content: `Error: ${msg}`, ts: Date.now() });
      })
      .finally(() => {
        if (!active) return;
        if (inFlightAreaKeyRef.current === selectedAreaKey) inFlightAreaKeyRef.current = null;
        setIsAreaAnalyzing(false);
      });

    return () => {
      active = false;
    };
  }, [selectedAreaKey]);

  useEffect(() => {
    if (!jobId) return;
    let mounted = true;
    const timer = window.setInterval(async () => {
      try {
        const status = await getManualTrainingStatus(jobId);
        if (!mounted) return;
        if (status.status === "completed" || status.status === "failed") {
          addMessage({
            role: "assistant",
            content: `Manual training ${status.status}. jobId=${status.jobId} log=${status.logPath}`,
            ts: Date.now(),
          });
          setJobId(null);
          window.clearInterval(timer);
        }
      } catch {
        
      }
    }, 5000);

    return () => {
      mounted = false;
      window.clearInterval(timer);
    };
  }, [jobId, addMessage]);

  const onSend = async () => {
    if (!input.trim() || !sessionId) return;
    const question = input.trim();
    const areaForChat = selectedArea ?? fallbackArea;
    setInput("");
    addMessage({ role: "user", content: question, ts: Date.now() });
    setIsChatLoading(true);

    try {
      const res = await askFollowup(sessionId, question, areaForChat);
      addMessage({ role: "assistant", content: res.reply, ts: Date.now() });
    } catch (err) {
      const msg = err instanceof Error ? err.message : String(err);
      addMessage({ role: "assistant", content: `Error: ${msg}`, ts: Date.now() });
    } finally {
      setIsChatLoading(false);
    }
  };

  const fileToDataUrl = async (file: File): Promise<string> =>
    new Promise((resolve, reject) => {
      const reader = new FileReader();
      reader.onload = () => resolve(String(reader.result || ""));
      reader.onerror = () => reject(new Error(`Failed to read ${file.name}`));
      reader.readAsDataURL(file);
    });

  const extractVideoFrames = async (file: File, maxFrames = 3): Promise<string[]> => {
    const objectUrl = URL.createObjectURL(file);
    const video = document.createElement("video");
    video.src = objectUrl;
    video.crossOrigin = "anonymous";
    video.muted = true;
    video.playsInline = true;

    await new Promise<void>((resolve, reject) => {
      video.onloadedmetadata = () => resolve();
      video.onerror = () => reject(new Error(`Failed to decode video ${file.name}`));
    });

    const duration = Math.max(0.5, video.duration || 0.5);
    const canvas = document.createElement("canvas");
    canvas.width = Math.max(320, Math.min(1024, video.videoWidth || 640));
    canvas.height = Math.max(240, Math.min(768, video.videoHeight || 360));
    const ctx = canvas.getContext("2d");
    if (!ctx) {
      URL.revokeObjectURL(objectUrl);
      throw new Error("Canvas context unavailable for video processing");
    }

    const frames: string[] = [];
    for (let i = 1; i <= maxFrames; i += 1) {
      const t = (duration * i) / (maxFrames + 1);
      await new Promise<void>((resolve) => {
        video.onseeked = () => {
          ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
          frames.push(canvas.toDataURL("image/jpeg", 0.85));
          resolve();
        };
        video.currentTime = Math.min(duration - 0.05, Math.max(0, t));
      });
    }

    URL.revokeObjectURL(objectUrl);
    return frames;
  };

  const onMediaSelect = async (files: FileList | null) => {
    if (!files || files.length === 0) return;
    setActionLoading(true);
    try {
      const newImages: string[] = [];
      const newFrames: string[] = [];
      const names: string[] = [];

      for (const file of Array.from(files)) {
        names.push(file.name);
        if (file.type.startsWith("image/")) {
          newImages.push(await fileToDataUrl(file));
        } else if (file.type.startsWith("video/")) {
          const frames = await extractVideoFrames(file, 3);
          newFrames.push(...frames);
        }
      }

      setMultimodalImages((prev) => [...prev, ...newImages].slice(0, 8));
      setMultimodalFrames((prev) => [...prev, ...newFrames].slice(0, 8));
      setMultimodalFilesLabel(names.join(", "));
    } catch (err) {
      const msg = err instanceof Error ? err.message : String(err);
      addMessage({ role: "assistant", content: `Media load error: ${msg}`, ts: Date.now() });
    } finally {
      setActionLoading(false);
    }
  };

  const onRunMultimodal = async () => {
    if (!multimodalQuestion.trim()) return;
    if (multimodalImages.length === 0 && multimodalFrames.length === 0 && !multimodalContext.trim() && !selectedArea) {
      addMessage({ role: "assistant", content: "Provide at least one input: map selection, text context, image, or video.", ts: Date.now() });
      return;
    }

    setIsMultimodalLoading(true);
    addMessage({ role: "user", content: multimodalQuestion.trim(), ts: Date.now() });
    try {
      const res = await runMultimodalResearch({
        sessionId: sessionId ?? undefined,
        question: multimodalQuestion.trim(),
        textContext: multimodalContext.trim() || undefined,
        area: selectedArea ?? undefined,
        imageDataUrls: multimodalImages.length ? multimodalImages : undefined,
        videoFramesDataUrls: multimodalFrames.length ? multimodalFrames : undefined,
      });

      setSessionId(res.sessionId);
      setAnalysis(res.analysis);
      if (selectedAreaKey) setCached(selectedAreaKey, res.analysis);
      addMessage({ role: "assistant", content: res.assistantMessage, ts: Date.now() });
      setMultimodalQuestion("");
    } catch (err) {
      const msg = err instanceof Error ? err.message : String(err);
      addMessage({ role: "assistant", content: `Multimodal error: ${msg}`, ts: Date.now() });
    } finally {
      setIsMultimodalLoading(false);
    }
  };

  const onSubmitFeedback = async () => {
    if (!selectedArea || !analysis) return;
    setActionLoading(true);
    try {
      const modelAnswer = [analysis.summary, analysis.environment, analysis.infrastructure, analysis.prediction].join("\n");
      const res = await submitFeedback({
        sessionId: sessionId ?? undefined,
        area: selectedArea,
        question: "Area analysis quality review",
        modelAnswer,
        researcherScore,
        metrics: { relevance, factuality, usefulness },
        notes: feedbackNotes || undefined,
        label: "researcher_manual_feedback",
      });
      addMessage({ role: "assistant", content: `Feedback logged. Reward=${res.reward}. Stored at ${res.feedbackPath}`, ts: Date.now() });
      setFeedbackNotes("");
    } catch (err) {
      const msg = err instanceof Error ? err.message : String(err);
      addMessage({ role: "assistant", content: `Feedback error: ${msg}`, ts: Date.now() });
    } finally {
      setActionLoading(false);
    }
  };

  const onManualTrain = async () => {
    if (!selectedArea) return;
    setActionLoading(true);
    try {
      const res = await startManualTraining({
        sessionId: sessionId ?? undefined,
        epochs,
        learningRate,
        batchSize,
        gradAccum,
        rewardWeight,
        targetMetric,
        minTargetValue,
      });
      setJobId(res.jobId);
      addMessage({ role: "assistant", content: `Manual training started. jobId=${res.jobId}. Status=${res.status}. Log=${res.logPath}`, ts: Date.now() });
    } catch (err) {
      const msg = err instanceof Error ? err.message : String(err);
      addMessage({ role: "assistant", content: `Manual train error: ${msg}`, ts: Date.now() });
    } finally {
      setActionLoading(false);
    }
  };

  return (
    <div className="flex h-full flex-col overflow-hidden rounded border border-[#1a1f2e] bg-[#0a0e14] font-['JetBrains_Mono']">
      <div className="flex items-center justify-between border-b border-[#1a1f2e] bg-[#0d1117] p-3">
        <div className="flex items-center gap-2">
          <span className="text-[0.7rem] font-bold tracking-widest text-[#00e5ff]">INTEL_ASSISTANT</span>
          <span className="h-1.5 w-1.5 rounded-full bg-[#00e676] animate-pulse"></span>
        </div>
        <div className="text-[0.6rem] text-[#4a5568]">
          {selectedArea ? `LOC: ${selectedArea.center.lat.toFixed(2)},${selectedArea.center.lng.toFixed(2)}` : "STANDBY"}
        </div>
      </div>

      <div className="flex-1 space-y-4 overflow-y-auto p-3 custom-scrollbar">
        {analysis && (
          <div className="space-y-3 rounded border border-[#1a1f2e] bg-[#0d1117] p-3 text-[0.72rem]">
            <div className="flex items-center justify-between border-b border-[#1a1f2e] pb-2">
              <span className="font-bold text-[#00e5ff]">AREA_SCAN_REPORT</span>
              <span className="rounded bg-[#00e676] px-1.5 py-0.5 text-[0.55rem] font-bold text-black uppercase">Complete</span>
            </div>
            
            <div className="space-y-2">
              <TechnicalSection title="Summary" content={analysis.summary} color="green" />
              <div className="grid grid-cols-2 gap-2">
                <TechnicalSection title="Demographics" content={analysis.demographics} />
                <TechnicalSection title="Economy" content={analysis.economy} />
              </div>
              <TechnicalSection title="Environment" content={analysis.environment} />
              <div className="grid grid-cols-2 gap-2">
                <TechnicalSection title="Infrastructure" content={analysis.infrastructure} />
                <TechnicalSection title="Historical Context" content={analysis.history} />
              </div>
              
              <div className="mt-4 rounded border border-[#3d0000] bg-[#1a0a0a] p-3">
                <div className="flex items-center gap-3">
                  <div className="flex h-10 w-10 items-center justify-center rounded-full border-2 border-[#ff1744] bg-[#1a0000] font-bold text-[#ff1744]">!</div>
                  <div>
                    <div className="text-[0.65rem] font-bold text-[#e0e0e0]">PREDICTIVE_INSIGHT</div>
                    <div className="text-[0.65rem] text-[#6b7280]">{analysis.prediction}</div>
                  </div>
                </div>
              </div>
            </div>
          </div>
        )}

        {/* Multimodal Section */}
        <div className="rounded border border-[#1a1f2e] bg-[#0a0a0a] p-3">
          <div className="mb-2 text-[0.65rem] font-bold uppercase tracking-widest text-[#ff9100]">MULTI-SENSOR_INPUT</div>
          <textarea
            value={multimodalContext}
            onChange={(e) => setMultimodalContext(e.target.value)}
            placeholder="ADD_SYSTEM_CONTEXT..."
            className="w-full rounded border border-[#1a1f2e] bg-[#0d1117] p-2 text-[0.7rem] text-[#c8ccd0] outline-none focus:border-[#00e5ff]"
            rows={2}
          />
          <div className="mt-2 flex items-center justify-between gap-2">
            <label className="flex cursor-pointer items-center gap-2 rounded border border-[#1a1f2e] bg-[#0d1117] px-2 py-1 text-[0.6rem] text-[#4a5568] hover:text-[#00e5ff]">
              <span>[UPLOAD_MEDIA]</span>
              <input
                type="file"
                accept="image/*,video/*"
                multiple
                onChange={(e) => onMediaSelect(e.target.files)}
                className="hidden"
              />
            </label>
            <span className="truncate text-[0.6rem] text-[#4a5568]">
              {multimodalFilesLabel || `IMG: ${multimodalImages.length} | VID: ${multimodalFrames.length}`}
            </span>
          </div>
          <div className="mt-2 flex gap-2">
            <input
              className="flex-1 rounded border border-[#1a1f2e] bg-[#0d1117] px-2 py-1 text-[0.7rem] text-[#c8ccd0] outline-none"
              placeholder="QUERY_MULTIMODAL_ENGINE..."
              value={multimodalQuestion}
              onChange={(e) => setMultimodalQuestion(e.target.value)}
            />
            <button
              onClick={onRunMultimodal}
              disabled={isMultimodalLoading || actionLoading}
              className="rounded bg-gradient-to-br from-[#00303f] to-[#004d40] px-3 py-1 text-[0.65rem] font-bold text-[#00e5ff] hover:from-[#004d40] hover:to-[#00695c] disabled:opacity-40"
            >
              RUN_SCAN
            </button>
          </div>
        </div>

        {/* Messages */}
        <div className="space-y-4">
          {messages.map((m, idx) => (
            <div key={`${m.ts}-${idx}`} className={`rounded border p-3 ${m.role === "user" ? "border-[#1a1f2e] bg-[#0d1a2d]" : "border-[#1a1f2e] bg-[#0d1117]"}`}>
              <div className={`mb-1 text-[0.6rem] font-bold uppercase tracking-widest ${m.role === "user" ? "text-[#00e5ff]" : "text-[#00e676]"}`}>
                {m.role === "user" ? "USER_INPUT" : "SYSTEM_OUTPUT"}
              </div>
              <div className="text-[0.75rem] leading-relaxed text-[#c8ccd0]">
                <ReactMarkdown>{m.content}</ReactMarkdown>
              </div>
            </div>
          ))}
        </div>

        {(isAreaAnalyzing || isChatLoading || isMultimodalLoading || actionLoading) && (
          <div className="flex items-center gap-2 rounded border border-[#1a1f2e] bg-[#0d1117] p-3 text-[0.7rem] text-[#4a5568]">
            <span className="h-1.5 w-1.5 animate-ping rounded-full bg-[#ff9100]"></span>
            PROCESSING_SIGNAL...
          </div>
        )}
      </div>

      <div className="border-t border-[#1a1f2e] bg-[#0d1117] p-3">
        <div className="flex gap-2">
          <input
            className="flex-1 rounded border border-[#1a1f2e] bg-[#0a0a0a] px-3 py-2 text-[0.75rem] text-[#c8ccd0] outline-none focus:border-[#00e5ff]"
            placeholder={canAsk ? "SEND_COMMAND..." : "INITIALIZE_SCAN_TO_PROCEED"}
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === "Enter") onSend();
            }}
            disabled={!canAsk || isChatLoading || isMultimodalLoading || actionLoading}
          />
          <button
            onClick={onSend}
            disabled={!canAsk || isChatLoading || isMultimodalLoading || actionLoading}
            className="rounded bg-gradient-to-br from-[#00303f] to-[#004d40] px-4 py-2 text-[0.7rem] font-bold text-[#00e5ff] hover:from-[#004d40] hover:to-[#00695c] disabled:opacity-40"
          >
            SEND
          </button>
        </div>
      </div>
      
      <style>{`
        .custom-scrollbar::-webkit-scrollbar { width: 4px; }
        .custom-scrollbar::-webkit-scrollbar-track { background: #050505; }
        .custom-scrollbar::-webkit-scrollbar-thumb { background: #1a1f2e; border-radius: 2px; }
        .custom-scrollbar::-webkit-scrollbar-thumb:hover { background: #00e5ff; }
      `}</style>
    </div>
  );
}

function TechnicalSection({ title, content, color = 'blue' }: { title: string; content: string; color?: 'blue' | 'green' | 'amber' }) {
  const colorClass = color === 'green' ? 'text-[#00e676]' : color === 'amber' ? 'text-[#ff9100]' : 'text-[#00e5ff]';
  return (
    <div className="rounded border border-[#1a1f2e]/50 bg-[#0a0a0a] p-2">
      <div className={`text-[0.6rem] font-bold uppercase tracking-widest ${colorClass} mb-1`}>{title}</div>
      <div className="text-[0.68rem] leading-relaxed text-[#c8ccd0]">{content}</div>
    </div>
  );
}

