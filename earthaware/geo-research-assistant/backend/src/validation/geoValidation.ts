import { z } from "zod";

export const coordinateSchema = z.tuple([z.number().min(-180).max(180), z.number().min(-90).max(90)]);

export const areaSchema = z.object({
  coordinates: z.array(coordinateSchema).min(1),
  center: z.object({
    lat: z.number().min(-90).max(90),
    lng: z.number().min(-180).max(180),
  }),
  areaType: z.enum(["point", "polygon"]),
  imageDataUrl: z
    .string()
    .max(22_000_000)
    .regex(/^data:image\/(png|jpeg|jpg);base64,/)
    .optional(),
  imageCaptureError: z.string().max(2000).optional(),
  sessionId: z.string().min(1).optional(),
});

const imageDataUrlSchema = z
  .string()
  .max(22_000_000)
  .regex(/^data:image\/(png|jpeg|jpg);base64,/);

export const chatSchema = z.object({
  sessionId: z.string().min(1),
  question: z.string().min(1).max(4000),
  area: areaSchema,
});

export const feedbackSchema = z.object({
  sessionId: z.string().min(1).optional(),
  area: areaSchema,
  question: z.string().min(1).max(4000),
  modelAnswer: z.string().min(1).max(12000),
  researcherScore: z.number().min(-1).max(1),
  metrics: z
    .object({
      relevance: z.number().min(0).max(1).optional(),
      factuality: z.number().min(0).max(1).optional(),
      usefulness: z.number().min(0).max(1).optional(),
    })
    .optional(),
  notes: z.string().max(4000).optional(),
  label: z.string().max(256).optional(),
});

export const manualTrainSchema = z.object({
  sessionId: z.string().min(1).optional(),
  epochs: z.number().int().min(1).max(25),
  learningRate: z.number().min(1e-6).max(1e-2),
  batchSize: z.number().int().min(1).max(8),
  gradAccum: z.number().int().min(1).max(64),
  rewardWeight: z.number().min(0).max(5),
  targetMetric: z.enum(["vqa_accuracy", "f1", "cider", "custom"]),
  minTargetValue: z.number().min(0).max(1),
  datasetPath: z.string().max(800).optional(),
  notes: z.string().max(4000).optional(),
});

export const multimodalSchema = z.object({
  sessionId: z.string().min(1).optional(),
  question: z.string().min(1).max(4000),
  textContext: z.string().max(30_000).optional(),
  area: areaSchema.optional(),
  imageDataUrls: z.array(imageDataUrlSchema).max(8).optional(),
  videoFramesDataUrls: z.array(imageDataUrlSchema).max(8).optional(),
});
