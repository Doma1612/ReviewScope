# Frontend

React 19 and TypeScript single page app, built with Vite. Server state is handled by TanStack Query, the cluster map by Plotly.

## Structure

| Path | Responsibility |
|---|---|
| `src/main.tsx` | Entry point, routes, query client |
| `src/api.ts` | Typed API client and response types; all requests send the auth cookie |
| `src/views/AuthPage.tsx` | Login and registration |
| `src/views/Dashboard.tsx` | Project grid, upload dialog with schema confirmation, live pipeline status |
| `src/views/DeckDashboard.tsx` | Alternative dashboard: project list next to the map of the selected project |
| `src/views/ProjectView.tsx` | Cluster map (2D/3D), cluster list, editing, document table, project chat |
| `src/views/ClusterDetail.tsx` | One cluster: summary, word cloud, rating distribution, documents, cluster chat |
| `src/views/PipelineView.tsx` | Pipeline steps of a project with the method behind each step |
| `src/views/SettingsView.tsx` | Rename, sharing with viewers, deletion |
| `src/ui/` | Shared components: app shell, theme, document table, word cloud, star rating, metrics panel, edit history, chat panel, badges |
| `src/optimistic.ts` | Optimistic cache updates for cluster edits, with rollback on error |
| `src/plot.ts` | Stratified point sampling that keeps the map responsive for large projects |
| `src/hover.ts`, `src/colors.ts` | Map tooltips and the cluster colour palette |
| `src/useSimulated.ts` | Detects whether the backend runs the simulated pipeline |

## Routes

| Route | View |
|---|---|
| `/login`, `/register` | Authentication |
| `/` | Project dashboard |
| `/deck` | Alternative dashboard |
| `/projects/:projectId` | Cluster view |
| `/projects/:projectId/clusters/:clusterId` | Cluster detail |
| `/projects/:projectId/pipeline` | Pipeline view |
| `/projects/:projectId/settings` | Project settings |

## Development

The Docker Compose stack runs the dev server with hot reload at http://localhost:5173. Without Docker, with the API running on port 8000:

```
npm install
npm run dev
```

`VITE_API_BASE_URL` sets the API address (default `http://localhost:8000`). The API accepts requests only from `http://localhost:5173`.

Type check and production build:

```
npm run build
```

The build reports a chunk size warning caused by Plotly; the build itself succeeds.
