import "dotenv/config";
import cors from "cors";
import express from "express";
import rateLimit from "express-rate-limit";
import researchRouter from "./routes/research.js";
import authRouter from "./routes/auth.js";

const app = express();
const port = Number(process.env.PORT || 8080);

app.use(express.json({ limit: "25mb" }));
app.use(
  cors({
    origin: process.env.CORS_ORIGIN || "http://localhost:5173",
  })
);

app.use(
  "/api/research",
  rateLimit({
    windowMs: 60_000,
    max: 30,
    standardHeaders: true,
    legacyHeaders: false,
  })
);

app.get("/health", (_req, res) => {
  res.json({ status: "ok", service: "geo-research-backend" });
});

app.use("/api/auth", authRouter);
app.use("/api/research", researchRouter);

app.listen(port, () => {
  console.log(`Geo research backend listening on http://localhost:${port}`);
});
