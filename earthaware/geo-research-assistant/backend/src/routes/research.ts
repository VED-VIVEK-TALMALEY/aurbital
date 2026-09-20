import { Router } from "express";
import { randomUUID } from "crypto";
import { promises as fs } from "fs";
import path from "path";
import { spawn } from "child_process";
import type { AreaPayload, GeoAnalysis, FeedbackPayload, ManualTrainPayload, MultimodalResearchPayload } from "../types.js";
import { analyzeAreaWithLLM, followupWithLLM, multimodalResearchWithLLM } from "../services/openaiService.js";
import { areaSchema, chatSchema, feedbackSchema, manualTrainSchema, multimodalSchema } from "../validation/geoValidation.js";
import { TTLCache } from "../utils/cache.js";

const router = Router();
const analysisCache = new TTLCache<GeoAnalysis>(1000 * 60 * 15);
const sessionHistory = new Map<string, { role: string; content: string }[]>();

const areaKey = (a: AreaPayload) => `${a.areaType}:${JSON.stringify(a.coordinates)}`;

type TrainJobStatus = "queued" | "running" | "completed" | "failed";

interface TrainJob {
  jobId: string;
  status: TrainJobStatus;
  startedAt: string;
  finishedAt?: string;
  command: string;
  logPath: string;
  config: ManualTrainPayload;
  exitCode?: number;
  error?: string;
}

const trainJobs = new Map<string, TrainJob>();

const backendRoot = process.cwd();
const earthawareRoot = path.resolve(backendRoot, "..", "..");
const artifactsDir = path.join(backendRoot, "runtime_artifacts");

async function ensureArtifactsDir() {
  await fs.mkdir(artifactsDir, { recursive: true });
}

async function appendJsonLine(filePath: string, payload: unknown) {
  await ensureArtifactsDir();
  const line = `${JSON.stringify(payload)}\n`;
  await fs.appendFile(filePath, line, "utf8");
}

async function readJsonFile<T>(filePath: string): Promise<T | null> {
  try {
    const raw = await fs.readFile(filePath, "utf8");
    return JSON.parse(raw) as T;
  } catch {
    return null;
  }
}

async function countJsonlLines(filePath: string): Promise<number> {
  try {
    const raw = await fs.readFile(filePath, "utf8");
    return raw.split("\n").map((l) => l.trim()).filter(Boolean).length;
  } catch {
    return 0;
  }
}

function safeRelativeOrAbsoluteDataset(inputPath?: string): string {
  if (!inputPath || !inputPath.trim()) {
    return path.join(earthawareRoot, "data", "training", "training_data_expanded.json");
  }
  if (path.isAbsolute(inputPath)) return inputPath;
  return path.join(earthawareRoot, inputPath);
}

router.post("/area", async (req, res) => {
  try {
    const payload = areaSchema.parse(req.body);
    const usingLocalModel = (process.env.AI_PROVIDER || (process.env.OPENAI_API_KEY ? "openai" : "local")).toLowerCase() === "local";
    if (usingLocalModel && payload.areaType === "polygon" && !payload.imageDataUrl) {
      res.status(400).json({
        error:
          "Polygon image capture missing. Use Shift+Drag and ensure capture succeeds before analysis.",
        details: payload.imageCaptureError || "No imageDataUrl in request payload.",
      });
      return;
    }
    const key = areaKey(payload);

    let analysis = analysisCache.get(key);
    if (!analysis) {
      analysis = await analyzeAreaWithLLM(payload);
      analysisCache.set(key, analysis);
    }

    const sessionId = payload.sessionId || randomUUID();
    const assistantMessage = [
      `### Geographic Overview\n${analysis.summary}`,
      `### Demographics\n${analysis.demographics}`,
      `### Economy\n${analysis.economy}`,
      `### Environment\n${analysis.environment}`,
      `### History\n${analysis.history}`,
      `### Infrastructure\n${analysis.infrastructure}`,
      `### Prediction\n${analysis.prediction}`,
    ].join("\n\n");

    const existing = sessionHistory.get(sessionId) ?? [];
    existing.push({ role: "assistant", content: assistantMessage });
    sessionHistory.set(sessionId, existing.slice(-20));

    res.json({ analysis, assistantMessage, sessionId });
  } catch (err) {
    const msg = err instanceof Error ? err.message : "Invalid request";
    res.status(400).json({ error: msg });
  }
});

router.post("/chat", async (req, res) => {
  try {
    const payload = chatSchema.parse(req.body);
    const history = sessionHistory.get(payload.sessionId) ?? [];
    history.push({ role: "user", content: payload.question });

    const reply = await followupWithLLM(payload.area, history, payload.question);
    history.push({ role: "assistant", content: reply });

    sessionHistory.set(payload.sessionId, history.slice(-30));
    res.json({ reply });
  } catch (err) {
    const msg = err instanceof Error ? err.message : "Invalid request";
    res.status(400).json({ error: msg });
  }
});

router.post("/multimodal", async (req, res) => {
  try {
    const payload = multimodalSchema.parse(req.body) as MultimodalResearchPayload;
    const analysis = await multimodalResearchWithLLM(payload);
    const sessionId = payload.sessionId || randomUUID();

    const assistantMessage = [
      `### Geographic Overview\n${analysis.summary}`,
      `### Demographics\n${analysis.demographics}`,
      `### Economy\n${analysis.economy}`,
      `### Environment\n${analysis.environment}`,
      `### History\n${analysis.history}`,
      `### Infrastructure\n${analysis.infrastructure}`,
      `### Prediction\n${analysis.prediction}`,
    ].join("\n\n");

    const existing = sessionHistory.get(sessionId) ?? [];
    existing.push({ role: "user", content: payload.question });
    existing.push({ role: "assistant", content: assistantMessage });
    sessionHistory.set(sessionId, existing.slice(-30));

    res.json({ analysis, assistantMessage, sessionId });
  } catch (err) {
    const msg = err instanceof Error ? err.message : "Invalid request";
    res.status(400).json({ error: msg });
  }
});

router.post("/feedback", async (req, res) => {
  try {
    const payload = feedbackSchema.parse(req.body) as FeedbackPayload;

    const rewardComponents = [
      payload.researcherScore,
      payload.metrics?.relevance,
      payload.metrics?.factuality,
      payload.metrics?.usefulness,
    ].filter((v): v is number => typeof v === "number");

    const reward =
      rewardComponents.length > 0
        ? Number((rewardComponents.reduce((a, b) => a + b, 0) / rewardComponents.length).toFixed(4))
        : Number(payload.researcherScore.toFixed(4));

    const record = {
      id: randomUUID(),
      createdAt: new Date().toISOString(),
      ...payload,
      reward,
    };

    const outPath = path.join(artifactsDir, "reinforcement_feedback.jsonl");
    await appendJsonLine(outPath, record);

    res.json({ success: true, reward, feedbackPath: outPath });
  } catch (err) {
    const msg = err instanceof Error ? err.message : "Invalid request";
    res.status(400).json({ error: msg });
  }
});

router.post("/manual-train", async (req, res) => {
  try {
    const payload = manualTrainSchema.parse(req.body) as ManualTrainPayload;
    await ensureArtifactsDir();

    const jobId = randomUUID();
    const datasetPath = safeRelativeOrAbsoluteDataset(payload.datasetPath);

    const pythonExe = path.join(earthawareRoot, "venv", "Scripts", "python.exe");
    const trainScript = path.join(earthawareRoot, "train_isro_eo_enhanced.py");
    const logPath = path.join(artifactsDir, `manual_train_${jobId}.log`);

    const args = [
      trainScript,
      "--data-path",
      datasetPath,
      "--epochs",
      String(payload.epochs),
      "--batch-size",
      String(payload.batchSize),
      "--grad-accum",
      String(payload.gradAccum),
      "--lr",
      String(payload.learningRate),
      "--class-balance",
    ];

    const outHandle = await fs.open(logPath, "a");
    const child = spawn(pythonExe, args, {
      cwd: earthawareRoot,
      detached: true,
      stdio: ["ignore", outHandle.fd, outHandle.fd],
    });

    const job: TrainJob = {
      jobId,
      status: "running",
      startedAt: new Date().toISOString(),
      command: `${pythonExe} ${args.join(" ")}`,
      logPath,
      config: payload,
    };

    trainJobs.set(jobId, job);

    child.on("close", async (code) => {
      const current = trainJobs.get(jobId);
      if (!current) return;
      current.status = code === 0 ? "completed" : "failed";
      current.exitCode = code ?? -1;
      current.finishedAt = new Date().toISOString();
      trainJobs.set(jobId, current);

      await appendJsonLine(path.join(artifactsDir, "manual_train_jobs.jsonl"), current);
      await outHandle.close();
    });

    child.on("error", async (error) => {
      const current = trainJobs.get(jobId);
      if (!current) return;
      current.status = "failed";
      current.error = error.message;
      current.finishedAt = new Date().toISOString();
      trainJobs.set(jobId, current);

      await appendJsonLine(path.join(artifactsDir, "manual_train_jobs.jsonl"), current);
      await outHandle.close();
    });

    child.unref();

    res.json({
      success: true,
      jobId,
      status: job.status,
      logPath,
      config: payload,
      note: "Manual training started. Poll /api/research/manual-train/:jobId for status.",
    });
  } catch (err) {
    const msg = err instanceof Error ? err.message : "Invalid request";
    res.status(400).json({ error: msg });
  }
});

router.get("/manual-train/:jobId", async (req, res) => {
  const { jobId } = req.params;
  const job = trainJobs.get(jobId);
  if (!job) {
    res.status(404).json({ error: "Training job not found" });
    return;
  }
  res.json(job);
});

router.get("/dashboard-metrics", async (_req, res) => {
  const metricsPath = path.join(earthawareRoot, "checkpoints", "isro_eo_enhanced_metrics.json");
  const day5Path = path.join(earthawareRoot, "results", "day5_evaluation.json");
  const baselinePath = path.join(earthawareRoot, "results", "baseline", "baseline_results.json");
  const feedbackPath = path.join(artifactsDir, "reinforcement_feedback.jsonl");

  const [trainMetrics, day5Metrics, baselineMetrics] = await Promise.all([
    readJsonFile<any>(metricsPath),
    readJsonFile<any>(day5Path),
    readJsonFile<any>(baselinePath),
  ]);
  const feedbackCount = await countJsonlLines(feedbackPath);

  const lastEpoch = Array.isArray(trainMetrics?.history) && trainMetrics.history.length
    ? trainMetrics.history[trainMetrics.history.length - 1]
    : null;

  res.json({
    training: {
      bestValLoss: trainMetrics?.best_val_loss ?? null,
      epochs: trainMetrics?.epochs ?? null,
      globalSteps: trainMetrics?.global_steps ?? null,
      lastEpoch: lastEpoch
        ? {
            epoch: lastEpoch.epoch ?? null,
            trainLoss: lastEpoch.train_loss ?? null,
            valLoss: lastEpoch.val_loss ?? null,
            lr: lastEpoch.lr ?? null,
          }
        : null,
    },
    evaluation: {
      day5: day5Metrics ?? null,
      baseline: baselineMetrics ?? null,
    },
    rl: {
      feedbackCount,
    },
  });
});

export default router;
