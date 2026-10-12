import type { EmbeddingPoint } from "./api";

export const POINT_CAP = 12000;

// A per-cluster floor prevents small clusters from disappearing during sampling.
export const STRATUM_FLOOR = 4;

export function samplePoints(points: EmbeddingPoint[]): EmbeddingPoint[] {
  if (points.length <= POINT_CAP) return points;

  const groups = new Map<string, EmbeddingPoint[]>();
  for (const point of points) {
    const key = point.cluster_id ?? "";
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
