# ReviewScope Frontend

React + TypeScript + Vite frontend for ReviewScope. It includes authentication screens, project dashboard, upload flow, spatial dashboard, pipeline process view, cluster visualization, cluster detail, and project settings.

## Requirements

- Node.js 22 recommended
- npm
- Backend API running at `http://localhost:8000`

## Environment

The frontend reads the API base URL from the root `.env` file:

```env
VITE_API_BASE_URL=http://localhost:8000
```

Do not commit `.env`. Use `.env.example` as the safe template.

## Run With Docker Compose

From the repository root:

```bash
docker compose up --build frontend
```

Usually, run the full stack instead:

```bash
docker compose up --build
```

Frontend URL:

```text
http://localhost:5173
```

## Local Frontend Development

From `src/frontend`:

```bash
npm install
npm run dev
```

Open:

```text
http://localhost:5173
```

Make sure the backend is running:

```text
http://localhost:8000/api/health
```

## Build

From `src/frontend`:

```bash
npm run build
```

Preview the production build:

```bash
npm run preview
```

## Main Routes

- `/login` — login
- `/register` — registration
- `/` — classic project dashboard
- `/deck` — spatial deck.gl-inspired dashboard
- `/projects/:projectId` — cluster visualization
- `/projects/:projectId/pipeline` — model/pipeline process view
- `/projects/:projectId/clusters/:clusterId` — cluster detail
- `/projects/:projectId/settings` — project settings and sharing

## Features

- httpOnly cookie authentication
- Project dashboard with upload modal
- CSV, JSON, and JSONL upload flow
- Pipeline status polling
- Methodology-based pipeline process page
- Plotly cluster visualization
- Deck-style spatial dashboard
- Dark and clear themes with `localStorage` persistence
- Project sharing/settings UI

## API Notes

The API client is in:

```text
src/frontend/src/api.ts
```

Requests use `credentials: "include"` because authentication is stored in an httpOnly cookie.

## Known Build Note

`npm run build` may show a chunk-size warning because Plotly is large. The build still succeeds. A later optimization can lazy-load Plotly or split visualization routes.
