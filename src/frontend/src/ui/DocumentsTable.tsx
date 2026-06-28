import { keepPreviousData, useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { useEffect, useMemo, useState } from "react";
import { Link } from "react-router-dom";

import { api, Cluster, DocumentFilter, SchemaColumn } from "../api";
import { applyReassign, beginOptimistic, invalidateAll, rollback, Snapshot } from "../optimistic";
import { captureClusters, toastReassign } from "../undo";

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

// Turn the per-column facet inputs into the typed filter list the API expects.
// Numeric/date columns contribute a gte (min/from) and/or lte (max/to); booleans an
// eq. Keys are `${column}:${bound}` where bound ∈ {min,max,from,to,eq}.
function buildFilters(columns: SchemaColumn[], values: Record<string, string>): DocumentFilter[] {
  const out: DocumentFilter[] = [];
  for (const col of columns) {
    if (col.type === "integer" || col.type === "float") {
      const min = values[`${col.name}:min`];
      const max = values[`${col.name}:max`];
      if (min) out.push({ column: col.name, op: "gte", value: min, type: col.type });
      if (max) out.push({ column: col.name, op: "lte", value: max, type: col.type });
    } else if (col.type === "date") {
      const from = values[`${col.name}:from`];
      const to = values[`${col.name}:to`];
      if (from) out.push({ column: col.name, op: "gte", value: from, type: "date" });
      if (to) out.push({ column: col.name, op: "lte", value: to, type: "date" });
    } else if (col.type === "boolean") {
      const eq = values[`${col.name}:eq`];
      if (eq) out.push({ column: col.name, op: "eq", value: eq, type: "boolean" });
    }
  }
  return out;
}

// One facet control, rendered per typed schema column.
function DocumentFacet({ column, values, onChange }: { column: SchemaColumn; values: Record<string, string>; onChange: (key: string, value: string) => void }) {
  if (column.type === "integer" || column.type === "float") {
    return (
      <span className="documents-facet">
        <label>{column.name}</label>
        <input type="number" placeholder="min" value={values[`${column.name}:min`] ?? ""} onChange={(event) => onChange(`${column.name}:min`, event.target.value)} />
        <input type="number" placeholder="max" value={values[`${column.name}:max`] ?? ""} onChange={(event) => onChange(`${column.name}:max`, event.target.value)} />
      </span>
    );
  }
  if (column.type === "date") {
    return (
      <span className="documents-facet">
        <label>{column.name}</label>
        <input type="date" value={values[`${column.name}:from`] ?? ""} onChange={(event) => onChange(`${column.name}:from`, event.target.value)} />
        <input type="date" value={values[`${column.name}:to`] ?? ""} onChange={(event) => onChange(`${column.name}:to`, event.target.value)} />
      </span>
    );
  }
  // boolean
  return (
    <span className="documents-facet">
      <label>{column.name}</label>
      <select value={values[`${column.name}:eq`] ?? ""} onChange={(event) => onChange(`${column.name}:eq`, event.target.value)}>
        <option value="">any</option>
        <option value="true">true</option>
        <option value="false">false</option>
      </select>
    </span>
  );
}

// Full document table whose columns come from the project schema. Used
// project-wide on ProjectView and cluster-scoped on ClusterDetail (via clusterId).
// Server-side pagination through GET /documents' limit/offset.
// When `editable` (owner + edit mode), each row gets a select-checkbox and a
// "move to cluster" dropdown, plus a bulk-reassign toolbar — wired to B3's single
// (reassignDocument) and bulk (bulkReassign) endpoints.
export function DocumentsTable({ projectId, clusterId, editable = false, clusters = [] }: { projectId: string; clusterId?: string; editable?: boolean; clusters?: Cluster[] }) {
  const [page, setPage] = useState(0);
  const [selected, setSelected] = useState<Set<string>>(new Set());
  const [bulkTarget, setBulkTarget] = useState("");
  // Facet filter values keyed by `${column}:${bound}` (bound = min/max/from/to/eq).
  const [facetValues, setFacetValues] = useState<Record<string, string>>({});
  const offset = page * PAGE_SIZE;
  const queryClient = useQueryClient();

  const schema = useQuery({ queryKey: ["schema", projectId], queryFn: () => api.getSchema(projectId), retry: false });

  // Build the typed filter list (R8) from the schema columns + facet inputs. Sent
  // to the server so filtering spans the whole dataset, not just the current page.
  const facetColumns = (schema.data?.columns ?? []).filter((col) => ["integer", "float", "date", "boolean"].includes(col.type));
  const filters = useMemo(() => buildFilters(facetColumns, facetValues), [facetColumns, facetValues]);
  const filtersKey = JSON.stringify(filters);

  const documents = useQuery({
    queryKey: ["documents", projectId, clusterId ?? null, page, filtersKey],
    queryFn: () => api.documents(projectId, { clusterId, limit: PAGE_SIZE, offset, filters }),
    placeholderData: keepPreviousData,
  });
  // Real total so the pager shows "of N" and "Next" never leads to an empty page.
  const count = useQuery({ queryKey: ["documents-count", projectId, clusterId ?? null, filtersKey], queryFn: () => api.documentsCount(projectId, { clusterId, filters }) });

  // Changing a facet can shrink the result below the current page — snap to page 0.
  useEffect(() => { setPage(0); }, [filtersKey]);

  // Look up a document's cluster for the per-row link (project-wide view only).
  const clusterById = new Map(clusters.map((cluster) => [cluster.id, cluster]));
  const showClusterColumn = !clusterId && clusters.length > 0;

  // Any reassignment changes membership; mutate the cache optimistically, roll back
  // on error, and reconcile via invalidateAll on settle. See ../optimistic.
  const onError = (_e: unknown, _v: unknown, ctx: { snapshot: Snapshot } | undefined) => {
    if (ctx) rollback(queryClient, ctx.snapshot);
  };
  const onSettled = () => invalidateAll(queryClient, projectId);
  const moveOne = useMutation({
    mutationFn: ({ id, target }: { id: string; target: string }) => api.reassignDocument(projectId, id, resolveTarget(target)),
    onMutate: async ({ id, target }: { id: string; target: string }) => {
      const snapshot = await beginOptimistic(queryClient, projectId);
      const prev = captureClusters(queryClient, projectId, [id], clusterId);
      applyReassign(queryClient, projectId, new Set([id]), resolveTarget(target), clusters);
      return { snapshot, prev };
    },
    onError,
    onSettled,
    onSuccess: (_data, _vars, ctx) => { if (ctx) toastReassign(queryClient, projectId, ctx.prev, 1); },
  });
  const moveBulk = useMutation({
    mutationFn: (target: string) => api.bulkReassign(projectId, [...selected], resolveTarget(target)),
    onMutate: async (target: string) => {
      const ids = [...selected];
      const snapshot = await beginOptimistic(queryClient, projectId);
      const prev = captureClusters(queryClient, projectId, ids, clusterId);
      applyReassign(queryClient, projectId, new Set(ids), resolveTarget(target), clusters);
      return { snapshot, prev, count: ids.length };
    },
    onError,
    onSettled,
    onSuccess: (_data, _vars, ctx) => { setSelected(new Set()); setBulkTarget(""); if (ctx) toastReassign(queryClient, projectId, ctx.prev, ctx.count); },
  });

  const rows = documents.data ?? [];
  // Prefer the saved schema; if none exists yet, derive columns from the first row.
  const columns = schema.data?.columns?.length
    ? schema.data.columns
    : Object.keys(rows[0]?.raw_data ?? {}).map((name) => ({ name, type: "text", is_primary_key: false }));
  const total = count.data?.total;
  // With a known total, "Next" is enabled only while more rows remain; fall back to
  // the page-full heuristic while the count is still loading.
  const hasNext = total != null ? offset + rows.length < total : rows.length === PAGE_SIZE;
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

  const setFacet = (key: string, value: string) =>
    setFacetValues((prev) => {
      const next = { ...prev };
      if (value) next[key] = value; else delete next[key];
      return next;
    });
  const hasActiveFilters = filters.length > 0;

  const facetsBar = facetColumns.length > 0 && (
    <div className="documents-facets">
      {facetColumns.map((col) => (
        <DocumentFacet key={col.name} column={col} values={facetValues} onChange={setFacet} />
      ))}
      {hasActiveFilters && <button className="documents-facets-clear" type="button" onClick={() => setFacetValues({})}>Clear filters</button>}
    </div>
  );

  const targetOptions = (
    <>
      <option value="">Move to…</option>
      {clusters.map((cluster) => <option key={cluster.id} value={cluster.id}>{cluster.label}</option>)}
      <option value={NOISE}>Noise</option>
    </>
  );

  if (documents.isLoading && !rows.length) return <div className="documents-table">{facetsBar}<p className="documents-table-empty">Loading documents…</p></div>;
  if (!rows.length) return <div className="documents-table">{facetsBar}<p className="documents-table-empty">{hasActiveFilters ? "No documents match these filters." : "No documents."}</p></div>;

  return (
    <div className="documents-table">
      {facetsBar}
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
              {showClusterColumn && <th>Cluster</th>}
              {editable && <th>Move</th>}
            </tr>
          </thead>
          <tbody>
            {rows.map((doc) => (
              <tr key={doc.id}>
                {editable && <td><input type="checkbox" checked={selected.has(doc.id)} onChange={() => toggleRow(doc.id)} aria-label="Select row" /></td>}
                {columns.map((col) => <td key={col.name}>{formatCell(doc.raw_data[col.name])}</td>)}
                {showClusterColumn && (
                  <td>
                    {doc.cluster_id && clusterById.has(doc.cluster_id)
                      ? <Link to={`/projects/${projectId}/clusters/${doc.cluster_id}`}>{clusterById.get(doc.cluster_id)!.label}</Link>
                      : <span className="documents-table-noise">noise</span>}
                  </td>
                )}
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
        <span className="documents-table-range">Showing {rows.length ? offset + 1 : offset}–{offset + rows.length}{total != null ? ` of ${total.toLocaleString()}` : ""}</span>
        <button className="button secondary" type="button" disabled={!hasNext} onClick={() => setPage((prev) => prev + 1)}>Next</button>
      </div>
    </div>
  );
}
