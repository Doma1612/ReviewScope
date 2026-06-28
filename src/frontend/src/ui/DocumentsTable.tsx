import { keepPreviousData, useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { useState } from "react";

import { api, Cluster } from "../api";
import { applyReassign, beginOptimistic, invalidateAll, rollback, Snapshot } from "../optimistic";

const PAGE_SIZE = 50;
// Sentinel select value for "noise" (Document.cluster_id IS NULL) — the API takes null.
const NOISE = "__noise__";
const resolveTarget = (value: string): string | null => (value === NOISE ? null : value);

// Render any raw_data cell value as a string; objects/arrays fall back to JSON.
function formatCell(value: unknown): string {
  if (value == null) return "";
  if (typeof value === "object") return JSON.stringify(value);
  return String(value);
}

// Full document table whose columns come from the project schema (B5). Used
// project-wide on ProjectView and cluster-scoped on ClusterDetail (via clusterId).
// Server-side pagination through GET /documents' limit/offset (F4).
// When `editable` (owner + edit mode, F5), each row gets a select-checkbox and a
// "move to cluster" dropdown, plus a bulk-reassign toolbar — wired to B3's single
// (reassignDocument) and bulk (bulkReassign) endpoints.
export function DocumentsTable({ projectId, clusterId, editable = false, clusters = [] }: { projectId: string; clusterId?: string; editable?: boolean; clusters?: Cluster[] }) {
  const [page, setPage] = useState(0);
  const [selected, setSelected] = useState<Set<string>>(new Set());
  const [bulkTarget, setBulkTarget] = useState("");
  const offset = page * PAGE_SIZE;
  const queryClient = useQueryClient();

  const schema = useQuery({ queryKey: ["schema", projectId], queryFn: () => api.getSchema(projectId), retry: false });
  const documents = useQuery({
    queryKey: ["documents", projectId, clusterId ?? null, page],
    queryFn: () => api.documents(projectId, { clusterId, limit: PAGE_SIZE, offset }),
    placeholderData: keepPreviousData,
  });

  // Any reassignment changes membership; mutate the cache optimistically, roll back
  // on error, and reconcile via invalidateAll on settle (F6). See ../optimistic.
  const onError = (_e: unknown, _v: unknown, ctx: { snapshot: Snapshot } | undefined) => {
    if (ctx) rollback(queryClient, ctx.snapshot);
  };
  const onSettled = () => invalidateAll(queryClient, projectId);
  const moveOne = useMutation({
    mutationFn: ({ id, target }: { id: string; target: string }) => api.reassignDocument(projectId, id, resolveTarget(target)),
    onMutate: async ({ id, target }: { id: string; target: string }) => {
      const snapshot = await beginOptimistic(queryClient, projectId);
      applyReassign(queryClient, projectId, new Set([id]), resolveTarget(target), clusters);
      return { snapshot };
    },
    onError,
    onSettled,
  });
  const moveBulk = useMutation({
    mutationFn: (target: string) => api.bulkReassign(projectId, [...selected], resolveTarget(target)),
    onMutate: async (target: string) => {
      const snapshot = await beginOptimistic(queryClient, projectId);
      applyReassign(queryClient, projectId, new Set(selected), resolveTarget(target), clusters);
      return { snapshot };
    },
    onError,
    onSettled,
    onSuccess: () => { setSelected(new Set()); setBulkTarget(""); },
  });

  const rows = documents.data ?? [];
  // Prefer the saved schema; if none exists yet, derive columns from the first row.
  const columns = schema.data?.columns?.length
    ? schema.data.columns
    : Object.keys(rows[0]?.raw_data ?? {}).map((name) => ({ name, type: "text", is_primary_key: false }));
  const hasNext = rows.length === PAGE_SIZE;
  const allOnPageSelected = rows.length > 0 && rows.every((doc) => selected.has(doc.id));

  const toggleRow = (id: string) =>
    setSelected((prev) => {
      const next = new Set(prev);
      if (next.has(id)) next.delete(id); else next.add(id);
      return next;
    });
  const toggleAll = () =>
    setSelected((prev) => {
      const next = new Set(prev);
      for (const doc of rows) { if (allOnPageSelected) next.delete(doc.id); else next.add(doc.id); }
      return next;
    });

  if (documents.isLoading) return <p className="documents-table-empty">Loading documents…</p>;
  if (!rows.length && page === 0) return <p className="documents-table-empty">No documents.</p>;

  const targetOptions = (
    <>
      <option value="">Move to…</option>
      {clusters.map((cluster) => <option key={cluster.id} value={cluster.id}>{cluster.label}</option>)}
      <option value={NOISE}>Noise</option>
    </>
  );

  return (
    <div className="documents-table">
      {editable && selected.size > 0 && (
        <div className="documents-table-bulk">
          <span>{selected.size} selected</span>
          <select value={bulkTarget} onChange={(event) => setBulkTarget(event.target.value)}>{targetOptions}</select>
          <button className="button" type="button" disabled={!bulkTarget || moveBulk.isPending} onClick={() => moveBulk.mutate(bulkTarget)}>Reassign</button>
          <button className="button secondary" type="button" onClick={() => setSelected(new Set())}>Clear</button>
        </div>
      )}
      <div className="documents-table-scroll">
        <table>
          <thead>
            <tr>
              {editable && <th><input type="checkbox" checked={allOnPageSelected} onChange={toggleAll} aria-label="Select all rows" /></th>}
              {columns.map((col) => <th key={col.name}>{col.name}{col.is_primary_key ? " 🔑" : ""}</th>)}
              {editable && <th>Move</th>}
            </tr>
          </thead>
          <tbody>
            {rows.map((doc) => (
              <tr key={doc.id}>
                {editable && <td><input type="checkbox" checked={selected.has(doc.id)} onChange={() => toggleRow(doc.id)} aria-label="Select row" /></td>}
                {columns.map((col) => <td key={col.name}>{formatCell(doc.raw_data[col.name])}</td>)}
                {editable && (
                  <td>
                    <select value="" onChange={(event) => { if (event.target.value) moveOne.mutate({ id: doc.id, target: event.target.value }); }}>{targetOptions}</select>
                  </td>
                )}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <div className="documents-table-pager">
        <button className="button secondary" type="button" disabled={page === 0} onClick={() => setPage((prev) => Math.max(0, prev - 1))}>Previous</button>
        <span className="documents-table-range">Showing {rows.length ? offset + 1 : offset}–{offset + rows.length}</span>
        <button className="button secondary" type="button" disabled={!hasNext} onClick={() => setPage((prev) => prev + 1)}>Next</button>
      </div>
    </div>
  );
}
