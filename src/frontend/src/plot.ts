import type { EmbeddingPoint } from "./api";

// F6 — WebGL/render cap. Above this many points the Plotly scatter (and the
// DeckDashboard DOM point cloud) is sampled down for display; the gap doc §3 uses
// the same ~12k threshold. Callers should also prefer scattergl (2D) when capped.
export const POINT_CAP = 12000;

// Deterministic stride sample to at most POINT_CAP points, preserving order so the
// sampled array index still maps back to a stable document (needed for click +
// lasso selection). Returns the original array untouched when under the cap.
export function samplePoints(points: EmbeddingPoint[]): EmbeddingPoint[] {
  if (points.length <= POINT_CAP) return points;
  const stride = points.length / POINT_CAP;
  const sampled: EmbeddingPoint[] = [];
  for (let i = 0; i < points.length && sampled.length < POINT_CAP; i += stride) {
    sampled.push(points[Math.floor(i)]);
  }
  return sampled;
}
