import type { QueryClient } from "@tanstack/react-query";

import { api, EmbeddingPoint } from "./api";
import { invalidateAll } from "./optimistic";
import { showToast } from "./toast";

// Snapshot each document's current cluster so a reassignment can be reversed. In a
// cluster-scoped table every row belongs to `clusterId`; project-wide we read the
// embeddings cache (the only place that holds every doc's current cluster).
export function captureClusters(
  qc: QueryClient,
  projectId: string,
  ids: string[],
  clusterId?: string,
): Map<string, string | null> {
  if (clusterId) return new Map(ids.map((id) => [id, clusterId]));
  const points = qc.getQueryData<EmbeddingPoint[]>(["embeddings", projectId]) ?? [];
  const byId = new Map(points.map((point) => [point.document_id, point.cluster_id]));
  return new Map(ids.map((id) => [id, byId.get(id) ?? null]));
}

// Move each document back to the cluster it came from, grouped so each original
// cluster is one bulk call. Then reconcile the cache.
export async function undoReassign(qc: QueryClient, projectId: string, prev: Map<string, string | null>): Promise<void> {
  const groups = new Map<string | null, string[]>();
  for (const [id, old] of prev) {
    const ids = groups.get(old) ?? [];
    ids.push(id);
    groups.set(old, ids);
  }
  for (const [old, ids] of groups) await api.bulkReassign(projectId, ids, old);
  invalidateAll(qc, projectId);
}

// Confirmation toast for a reassignment, with an inline Undo that puts every moved
// document back where it was.
export function toastReassign(qc: QueryClient, projectId: string, prev: Map<string, string | null>, count: number): void {
  showToast({
    message: `Moved ${count} document${count === 1 ? "" : "s"}`,
    actionLabel: "Undo",
    onAction: () => { void undoReassign(qc, projectId, prev); },
  });
}
