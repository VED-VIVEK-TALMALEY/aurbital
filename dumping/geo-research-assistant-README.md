# Geospatial AI Research Assistant

This module adds a production-style full-stack geospatial research app:
- 3D live map (MapLibre GL JS + WebGL, no paid token required)
- Point/polygon selection with geometry state
- AI geospatial analyst chat with context memory
- Node/Express backend with OpenAI integration, validation, caching, and rate limiting

## Folder Structure

```text
geo-research-assistant/
  frontend/
    src/
      components/
        Map3D.tsx
        ChatPanel.tsx
      store/
        useGeoStore.ts
      lib/
        api.ts
      App.tsx
      main.tsx
      styles.css
      types.ts
    package.json
    tailwind.config.ts
    postcss.config.js
    tsconfig.json
    vite.config.ts
    .env.example
  backend/
    src/
      routes/research.ts
      services/openaiService.ts
      validation/geoValidation.ts
      utils/cache.ts
      server.ts
      types.ts
    package.json
    tsconfig.json
    .env.example
```

## Frontend Setup

```bash
cd frontend
npm install
cp .env.example .env
# set VITE_MAP_STYLE_URL and VITE_API_BASE_URL
npm run dev
```

## Backend Setup

```bash
cd backend
npm install
cp .env.example .env
# set AI_PROVIDER=local and LOCAL_MODEL_API_BASE=http://localhost:8000 to use your EarthAware model
# or set OPENAI_API_KEY and optional OPENAI_MODEL for OpenAI
npm run dev
```

## Runtime URLs

- Frontend: http://localhost:5173
- Backend: http://localhost:8080

## API Flow

1. User clicks map or draws polygon
2. Frontend sends selected geometry:
```json
{
  "coordinates": [[77.5,12.9],[77.6,12.9],[77.6,13.0],[77.5,13.0],[77.5,12.9]],
  "center": {"lat": 12.95, "lng": 77.55},
  "areaType": "polygon"
}
```
3. Backend validates and calls OpenAI
4. Returns structured analysis + assistant message

## Deployment Notes

- Frontend: Vercel/Netlify (set `VITE_*` env vars)
- Backend: Render/Fly.io/Railway (set `OPENAI_API_KEY`)
- Enable HTTPS in production
- Tighten CORS origin list before deploy

## Prompt Template Used

Backend uses this structured instruction:

You are a geospatial research analyst.

Analyze the selected geographic region using the following coordinates:
[COORDINATES]

Provide structured analysis including:
- Geographic Overview
- Demographics
- Economic Activity
- Environmental Factors
- Infrastructure
- Historical or Strategic Importance

Respond in clear research report format suitable for urban planning or business analysis.


