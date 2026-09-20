import { Router } from "express";

const router = Router();

const AUTH_USER = process.env.AUTH_USER || "researcher";
const AUTH_PASSWORD = process.env.AUTH_PASSWORD || "earthaware123";
const AUTH_TOKEN = process.env.AUTH_TOKEN || "earthaware-local-token";

router.post("/login", (req, res) => {
  const username = String(req.body?.username || "");
  const password = String(req.body?.password || "");

  if (username !== AUTH_USER || password !== AUTH_PASSWORD) {
    res.status(401).json({ error: "Invalid credentials" });
    return;
  }

  res.json({
    token: AUTH_TOKEN,
    user: { username: AUTH_USER, role: "research_admin" },
  });
});

router.get("/me", (req, res) => {
  const token = String(req.headers.authorization || "").replace(/^Bearer\s+/i, "");
  if (!token || token !== AUTH_TOKEN) {
    res.status(401).json({ error: "Unauthorized" });
    return;
  }
  res.json({ user: { username: AUTH_USER, role: "research_admin" } });
});

export default router;
