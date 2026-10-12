import type { QueryClient } from "@tanstack/react-query";

import type { Cluster, DocumentItem, EmbeddingPoint } from "./api";


// Prefix keys include every paginated query for the project.
const affectedKeys = (projectId: string): readonly unknown[][] => [
  ["clusters", projectId],
  ["embeddings", projectId],
  ["documents", projectId],
  ["cluster", projectId],
  ["cluster-docs", projectId],
  ["edits", projectId],
];

export type Snapshot = Array<[readonly unknown[], unknown]>;

// Cancel refetches before optimistic writes so late responses cannot overwrite them.
export async function beginOptimistic(qc: QueryClient, projectId: string): Promise<Snapshot> {
  const keys = affectedKeys(projectId);
  await Promise.all(keys.map((queryKey) => qc.cancelQueries({ queryKey })));
  const snapshot: Snapshot = [];
  for (const queryKey of keys) {
    for (const [key, data] of qc.getQueriesData({ queryKey })) snapshot.push([key, data]);
  }
  return snapshot;
}

export function rollback(qc: QueryClient, snapshot: Snapshot): void {
  for (const [key, data] of snapshot) qc.setQueryData(key, data);
}

export function invalidateAll(qc: QueryClient, projectId: string): void {
  for (const queryKey of affectedKeys(projectId)) qc.invalidateQueries({ queryKey });
}

const patchDocs =
  (predicate: (doc: DocumentItem) => boolean, clusterId: string | null) =>
  (docs?: DocumentItem[]) =>
    docs?.map((doc) => (predicate(doc) ? { ...doc, cluster_id: clusterId } : doc));

export function applyRename(qc: QueryClient, projectId: string, clusterId: string, label: string): void {
  qc.setQueriesData<Cluster[]>({ queryKey: ["clusters", projectId] }, (cs) =>
    cs?.map((c) => (c.id === clusterId ? { ...c, label, label_source: "hitl_override" } : c)),
  );
  qc.setQueryData<Cluster>(["cluster", projectId, clusterId], (c) =>
    c ? { ...c, label, label_source: "hitl_override" } : c,
  );
}

export function applyMerge(
  qc: QueryClient,
  projectId: string,
  sourceIds: string[],
  targetId: string,
  clusters: Cluster[],
): void {
  const sources = new Set(sourceIds);
  const targetLabel = clusters.find((c) => c.id === targetId)?.label ?? null;

  qc.setQueriesData<EmbeddingPoint[]>({ queryKey: ["embeddings", projectId] }, (points) =>
    points?.map((point) =>
      point.cluster_id && sources.has(point.cluster_id)
        ? { ...point, cluster_id: targetId, cluster_label: targetLabel }
        : point,
    ),
  );
  qc.setQueriesData<Cluster[]>({ queryKey: ["clusters", projectId] }, (cs) => {
    if (!cs) return cs;
    const moved = cs.filter((c) => sources.has(c.id)).reduce((sum, c) => sum + c.size, 0);
    return cs.filter((c) => !sources.has(c.id)).map((c) => (c.id === targetId ? { ...c, size: c.size + moved } : c));
  });
  const patch = patchDocs((doc) => Boolean(doc.cluster_id && sources.has(doc.cluster_id)), targetId);
  qc.setQueriesData<DocumentItem[]>({ queryKey: ["documents", projectId] }, patch);
  qc.setQueriesData<DocumentItem[]>({ queryKey: ["cluster-docs", projectId] }, patch);
}

export function applyJunk(qc: QueryClient, projectId: string, clusterId: string): void {
  qc.setQueriesData<EmbeddingPoint[]>({ queryKey: ["embeddings", projectId] }, (points) =>
    points?.map((point) =>
      point.cluster_id === clusterId ? { ...point, cluster_id: null, cluster_label: null } : point,
    ),
  );
  qc.setQueriesData<Cluster[]>({ queryKey: ["clusters", projectId] }, (cs) => cs?.filter((c) => c.id !== clusterId));
  const patch = patchDocs((doc) => doc.cluster_id === clusterId, null);
  qc.setQueriesData<DocumentItem[]>({ queryKey: ["documents", projectId] }, patch);
  qc.setQueriesData<DocumentItem[]>({ queryKey: ["cluster-docs", projectId] }, patch);
}
