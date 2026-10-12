import { keepPreviousData, useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { useEffect, useMemo, useState } from "react";
import { Link } from "react-router-dom";

import { api, Cluster, DocumentFilter, SchemaColumn } from "../api";
import { invalidateAll } from "../optimistic";

const PAGE_SIZE = 50;
const NOISE = "__noise__";
const resolveTarget = (value: string): string | null => (value === NOISE ? null : value);

function formatCell(value: unknown): string {
  if (value == null) return "";
  if (typeof value === "object") return JSON.stringify(value);
  return String(value);
}

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

export function DocumentsTable({ projectId, clusterId, editable = false, clusters = [], sentence = false }: { projectId: string; clusterId?: string; editable?: boolean; clusters?: Cluster[]; sentence?: boolean }) {
  const [page, setPage] = useState(0);
  const [selected, setSelected] = useState<Set<string>>(new Set());
  const [bulkTarget, setBulkTarget] = useState("");
  const [showIds, setShowIds] = useState(false);
  const [facetValues, setFacetValues] = useState<Record<string, string>>({});
  const offset = page * PAGE_SIZE;
  const queryClient = useQueryClient();

  const schema = useQuery({ queryKey: ["schema", projectId], queryFn: () => api.getSchema(projectId), retry: false });

  const facetColumns = (schema.data?.columns ?? []).filter((col) => ["integer", "float", "date", "boolean"].includes(col.type));
  const filters = useMemo(() => buildFilters(facetColumns, facetValues), [facetColumns, facetValues]);
  const filtersKey = JSON.stringify(filters);

  const documents = useQuery({
    queryKey: ["documents", projectId, clusterId ?? null, page, filtersKey],
    queryFn: () => api.documents(projectId, { clusterId, limit: PAGE_SIZE, offset, filters }),
    placeholderData: keepPreviousData,
  });
  const count = useQuery({ queryKey: ["documents-count", projectId, clusterId ?? null, filtersKey], queryFn: () => api.documentsCount(projectId, { clusterId, filters }) });

  useEffect(() => { setPage(0); }, [filtersKey]);

  const clusterById = new Map(clusters.map((cluster) => [cluster.id, cluster]));
  const showClusterColumn = !clusterId && clusters.length > 0;

  // Review moves affect multiple mentions; invalidate rather than patch document-keyed caches.
  const onSettled = () => invalidateAll(queryClient, projectId);
  const moveOne = useMutation({
    mutationFn: ({ id, target }: { id: string; target: string }) => api.reassignReview(projectId, id, resolveTarget(target)),
    onSettled,
  });
  const moveBulk = useMutation({
    mutationFn: (target: string) => api.bulkReassign(projectId, [...selected], resolveTarget(target)),
    onSettled,
    onSuccess: () => { setSelected(new Set()); setBulkTarget(""); },
  });

  const rows = documents.data ?? [];
  const allColumns = schema.data?.columns?.length
    ? schema.data.columns
    : Object.keys(rows[0]?.raw_data ?? {}).map((name) => ({ name, type: "text", is_primary_key: false }));
  const idColumns = allColumns.filter((col) => col.is_primary_key || /^id$|_id$/i.test(col.name));
  const columns = showIds || idColumns.length === allColumns.length
    ? allColumns
    : allColumns.filter((col) => !idColumns.includes(col));
  const total = count.data?.total;
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

  const idToggle = idColumns.length > 0 && idColumns.length < allColumns.length && (
    <label className="documents-facets-ids">
      <input type="checkbox" checked={showIds} onChange={(event) => setShowIds(event.target.checked)} />
      Show IDs
    </label>
  );

  const facetsBar = (facetColumns.length > 0 || idToggle) && (
    <div className="documents-facets">
      {facetColumns.map((col) => (
        <DocumentFacet key={col.name} column={col} values={facetValues} onChange={setFacet} />
      ))}
      {idToggle}
      {hasActiveFilters && <button className="documents-facets-clear" type="button" onClick={() => setFacetValues({})}>Clear filters</button>}
    </div>
  );

  const targetOptions = (
    <>
      <option value="">{sentence ? "Move all to…" : "Move to…"}</option>
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
              {showClusterColumn && <th>{sentence ? "Clusters" : "Cluster"}</th>}
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
                    {sentence ? (
                      doc.memberships && doc.memberships.length > 0 ? (
                        <span className="membership-chips">
                          {doc.memberships.map((m) => (
                            <Link
                              key={m.cluster_id}
                              className={`membership-chip ${m.cluster_id === doc.primary_cluster_id ? "primary" : ""}`}
                              to={`/projects/${projectId}/clusters/${m.cluster_id}`}
                            >
                              {m.cluster_label}<small>×{m.mention_count}</small>
                            </Link>
                          ))}
                        </span>
                      ) : <span className="documents-table-noise">noise</span>
                    ) : (
                      doc.cluster_id && clusterById.has(doc.cluster_id)
                        ? <Link to={`/projects/${projectId}/clusters/${doc.cluster_id}`}>{clusterById.get(doc.cluster_id)!.label}</Link>
                        : <span className="documents-table-noise">noise</span>
                    )}
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
