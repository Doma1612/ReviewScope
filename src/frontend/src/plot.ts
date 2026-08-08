import type { EmbeddingPoint } from "./api";

// WebGL/render cap. Above this many points the Plotly scatter (and the
// DeckDashboard DOM point cloud) is sampled down for display at a ~12k threshold.
// Callers should also prefer scattergl (2D) when capped.
export const POINT_CAP = 12000;

// Minimum points kept per cluster when sampling. A small cluster that is far
// below its proportional quota would vanish under a flat stride; this floor keeps
// it visible on the overview map (capped at the cluster's actual size).
export const STRATUM_FLOOR = 4;

// Cluster-stratified deterministic sample to ~POINT_CAP points. A flat stride
// can sample a small cluster down to zero points, making it silently disappear from
// the map — so we sample *within each cluster* instead: every cluster gets its
// proportional share but never fewer than STRATUM_FLOOR points (noise = its own
// stratum). The per-cluster sample is a stride sample, so it stays deterministic
// and the returned array index still maps back to a stable document (needed for
// click + lasso). Returns the original array untouched when under the cap.
export function samplePoints(points: EmbeddingPoint[]): EmbeddingPoint[] {
  if (points.length <= POINT_CAP) return points;

  const groups = new Map<string, EmbeddingPoint[]>();
  for (const point of points) {
    const key = point.cluster_id ?? ""; // noise → "" stratum
    const group = groups.get(key);
    if (group) group.push(point);
    else groups.set(key, [point]);
  }

  const total = points.length;
  const sampled: EmbeddingPoint[] = [];
  for (const group of groups.values()) {
    const proportional = Math.round((group.length / total) * POINT_CAP);
    const quota = Math.min(group.length, Math.max(STRATUM_FLOOR, proportional));
    if (quota >= group.length) {
      for (const point of group) sampled.push(point);
      continue;
    }
    const stride = group.length / quota;
    for (let i = 0, taken = 0; i < group.length && taken < quota; i += stride, taken += 1) {
      sampled.push(group[Math.floor(i)]);
    }
  }
  return sampled;
}
