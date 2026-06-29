import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { useEffect, useMemo, useRef, useState } from "react";
import Plot from "react-plotly.js";
import { Link, useParams } from "react-router-dom";

import { api, Cluster, ClusterEdit } from "../api";
import { NOISE_COLOR, clusterColor, pointColor } from "../colors";
import { hoverHtml, sentimentSummary } from "../hover";
import {
  applyCreateFromSelection,
  applyJunk,
  applyMerge,
  applyRename,
  applyReassign,
  beginOptimistic,
  invalidateAll,
  rollback,
  Snapshot,
} from "../optimistic";
import { POINT_CAP, samplePoints } from "../plot";
import { showToast } from "../toast";
import { captureClusters, toastReassign } from "../undo";
import { CohesionChip } from "../ui/CohesionChip";
import { DocumentsTable } from "../ui/DocumentsTable";
import { EditHistory } from "../ui/EditHistory";
import { LabelSourceBadge } from "../ui/LabelSourceBadge";
import { MetricsPanel } from "../ui/MetricsPanel";
import { StarRating } from "../ui/StarRating";
import { WordCloud } from "../ui/WordCloud";

// Sentinel select value for "noise" (cluster_id = null); the API takes null.
const NOISE = "__noise__";
const resolveTarget = (value: string): string | null => (value === NOISE ? null : value);

// Undo: a bulk_reassign edit stores a per-doc `before` map (doc id → old
// cluster id, null = noise). Group those docs by their original cluster so we can
// move each group back in one bulkReassign call.
const beforeGroups = (payload: Record<string, unknown>): [string | null, string[]][] => {
  const before = (payload.before ?? {}) as Record<string, string | null>;
  const groups = new Map<string | null, string[]>();
  for (const [docId, oldId] of Object.entries(before)) {
    const ids = groups.get(oldId) ?? [];
    ids.push(docId);
    groups.set(oldId, ids);
  }
  return [...groups];
};

export function ProjectView() {
  const { projectId = "" } = useParams();
  const queryClient = useQueryClient();
  const project = useQuery({ queryKey: ["project", projectId], queryFn: () => api.project(projectId) });
  const status = useQuery({ queryKey: ["pipeline", projectId], queryFn: () => api.pipelineStatus(projectId), refetchInterval: project.data?.status === "ready" ? false : 3000 });
  const embeddings = useQuery({ queryKey: ["embeddings", projectId], queryFn: () => api.embeddings(projectId), enabled: project.data?.status === "ready" });
  const clusters = useQuery({ queryKey: ["clusters", projectId], queryFn: () => api.clusters(projectId), enabled: project.data?.status === "ready" });

  const [mode, setMode] = useState<"3d" | "2d">("3d");
  const [showDocuments, setShowDocuments] = useState(false);
  const [highlightedClusterId, setHighlightedClusterId] = useState<string | null>(null);
  const cardRefs = useRef<Record<string, HTMLElement | null>>({});

  // Cluster-list controls: sort key + free-text label filter.
  const [clusterSort, setClusterSort] = useState<"size" | "sentiment" | "cohesion" | "label">("size");
  const [clusterFilter, setClusterFilter] = useState("");

  // Editing state (owner only).
  const isOwner = project.data?.role === "owner";
  const [editMode, setEditMode] = useState(false);
  const [selectedPointIds, setSelectedPointIds] = useState<string[]>([]); // lasso selection → document ids
  const [reassignTarget, setReassignTarget] = useState("");
  const [newClusterLabel, setNewClusterLabel] = useState("");
  const [selectedClusterIds, setSelectedClusterIds] = useState<Set<string>>(new Set());
  const [mergeTarget, setMergeTarget] = useState("");

  // Reset the highlight + edit state when switching projects (mirrors DeckDashboard).
  useEffect(() => {
    setHighlightedClusterId(null);
    setEditMode(false);
    setSelectedPointIds([]);
    setSelectedClusterIds(new Set());
  }, [projectId]);

  // Leaving edit mode drops any in-flight selections so the read-only view stays clean.
  useEffect(() => {
    if (!editMode) { setSelectedPointIds([]); setSelectedClusterIds(new Set()); }
  }, [editMode]);

  // Scroll the highlighted cluster's card into view when it changes.
  useEffect(() => {
    if (highlightedClusterId) cardRefs.current[highlightedClusterId]?.scrollIntoView({ behavior: "smooth", block: "nearest" });
  }, [highlightedClusterId]);

  // Structural edits change membership/aggregates. Each mutation writes its change
  // into the cache optimistically (onMutate), rolls back on error, and reconciles
  // with the server's recompute via invalidateAll (onSettled). See ../optimistic.
  const onError = (_e: unknown, _v: unknown, ctx: { snapshot: Snapshot } | undefined) => {
    if (ctx) rollback(queryClient, ctx.snapshot);
  };
  const onSettled = () => invalidateAll(queryClient, projectId);

  const bulkReassignM = useMutation({
    mutationFn: (target: string) => api.bulkReassign(projectId, selectedPointIds, resolveTarget(target)),
    onMutate: async (target: string) => {
      const ids = [...selectedPointIds];
      const snapshot = await beginOptimistic(queryClient, projectId);
      const prev = captureClusters(queryClient, projectId, ids);
      applyReassign(queryClient, projectId, new Set(ids), resolveTarget(target), clusters.data ?? []);
      return { snapshot, prev, count: ids.length };
    },
    onError,
    onSettled,
    onSuccess: (_data, _vars, ctx) => { setSelectedPointIds([]); setReassignTarget(""); if (ctx) toastReassign(queryClient, projectId, ctx.prev, ctx.count); },
  });
  const createFromSelectionM = useMutation({
    mutationFn: (label: string) => api.createClusterFromSelection(projectId, selectedPointIds, label),
    onMutate: async (label: string) => {
      const snapshot = await beginOptimistic(queryClient, projectId);
      applyCreateFromSelection(queryClient, projectId, new Set(selectedPointIds), label);
      return { snapshot };
    },
    onError,
    onSettled,
    onSuccess: () => { setSelectedPointIds([]); setNewClusterLabel(""); },
  });
  const renameM = useMutation({
    mutationFn: ({ id, label }: { id: string; label: string; prevLabel: string }) => api.updateCluster(projectId, id, { label }),
    onMutate: async ({ id, label }: { id: string; label: string; prevLabel: string }) => {
      const snapshot = await beginOptimistic(queryClient, projectId);
      applyRename(queryClient, projectId, id, label);
      return { snapshot };
    },
    onError,
    onSettled,
    onSuccess: (_data, { id, label, prevLabel }) => {
      showToast({
        message: `Renamed to "${label}"`,
        actionLabel: "Undo",
        onAction: () => { void api.updateCluster(projectId, id, { label: prevLabel }).then(() => invalidateAll(queryClient, projectId)); },
      });
    },
  });
  const mergeM = useMutation({
    mutationFn: ({ sources, target }: { sources: string[]; target: string }) => api.mergeClusters(projectId, sources, target),
    onMutate: async ({ sources, target }: { sources: string[]; target: string }) => {
      const snapshot = await beginOptimistic(queryClient, projectId);
      applyMerge(queryClient, projectId, sources, target, clusters.data ?? []);
      return { snapshot };
    },
    onError,
    onSettled,
    onSuccess: () => { setSelectedClusterIds(new Set()); setMergeTarget(""); },
  });
  const junkM = useMutation({
    mutationFn: (id: string) => api.deleteCluster(projectId, id),
    onMutate: async (id: string) => {
      const snapshot = await beginOptimistic(queryClient, projectId);
      applyJunk(queryClient, projectId, id);
      return { snapshot };
    },
    onError,
    onSettled,
  });

  // Invert a reversible edit. Reuses the existing mutation endpoints (so the
  // undo itself is recorded as a fresh edit) and the same optimistic cache apply.
  const undoM = useMutation({
    mutationFn: async (edit: ClusterEdit) => {
      if (edit.action === "reassign_doc" && edit.document_id) {
        await api.reassignDocument(projectId, edit.document_id, edit.cluster_id);
      } else if (edit.action === "rename_label" && edit.cluster_id) {
        await api.updateCluster(projectId, edit.cluster_id, { label: String(edit.payload.before) });
      } else if (edit.action === "bulk_reassign") {
        for (const [oldId, ids] of beforeGroups(edit.payload)) await api.bulkReassign(projectId, ids, oldId);
      }
    },
    onMutate: async (edit: ClusterEdit) => {
      const snapshot = await beginOptimistic(queryClient, projectId);
      const cs = clusters.data ?? [];
      if (edit.action === "reassign_doc" && edit.document_id) {
        applyReassign(queryClient, projectId, new Set([edit.document_id]), edit.cluster_id, cs);
      } else if (edit.action === "rename_label" && edit.cluster_id) {
        applyRename(queryClient, projectId, edit.cluster_id, String(edit.payload.before));
      } else if (edit.action === "bulk_reassign") {
        for (const [oldId, ids] of beforeGroups(edit.payload)) applyReassign(queryClient, projectId, new Set(ids), oldId, cs);
      }
      return { snapshot };
    },
    onError,
    onSettled,
  });

  const toggleClusterSelect = (id: string) =>
    setSelectedClusterIds((prev) => {
      const next = new Set(prev);
      if (next.has(id)) next.delete(id); else next.add(id);
      return next;
    });
  const handleMergeSelected = () => {
    const ids = [...selectedClusterIds];
    if (ids.length < 2) return;
    const target = mergeTarget && ids.includes(mergeTarget) ? mergeTarget : ids[0];
    mergeM.mutate({ sources: ids.filter((id) => id !== target), target });
  };

  const points = embeddings.data ?? [];
  // WebGL/render cap: sample huge projects down for display and force 2D
  // scattergl, which renders far more smoothly than scatter3d at this scale.
  const capped = points.length > POINT_CAP;
  const displayPoints = useMemo(() => samplePoints(points), [points]);
  const webglMode = capped ? "2d" : mode;
  const clusterIndex = new Map((clusters.data ?? []).map((cluster, index) => [cluster.id, index]));
  const colors = displayPoints.map((point) => {
    const base = pointColor(point.cluster_id, clusterIndex);
    const muted = highlightedClusterId && point.cluster_id !== highlightedClusterId;
    return muted ? `${base}22` : base;
  });
  // Lasso-reassign is disabled on the capped overview: the lasso only sees the
  // sampled points, so reassigning would silently edit a subset of the region while
  // implying the whole region. Bulk-reassign on the exact document table instead.
  const lassoEnabled = editMode && webglMode === "2d" && !capped;
  // Noise = documents left unclustered (cluster_id null). Counted from the full
  // embeddings set, not the sampled display set, so the figure is honest.
  const noiseCount = points.filter((point) => point.cluster_id == null).length;

  // Cluster list: apply the label filter + chosen sort. Card colors are keyed by
  // cluster id (clusterIndex), so reordering the display list never recolors points.
  const visibleClusters = useMemo(() => {
    const needle = clusterFilter.trim().toLowerCase();
    const list = (clusters.data ?? []).filter((cluster) => cluster.label.toLowerCase().includes(needle));
    const num = (value: number | null) => (value == null ? -Infinity : value);
    const sorted = [...list];
    if (clusterSort === "size") sorted.sort((a, b) => b.size - a.size);
    else if (clusterSort === "sentiment") sorted.sort((a, b) => num(b.sentiment_avg) - num(a.sentiment_avg));
    else if (clusterSort === "cohesion") sorted.sort((a, b) => num(b.cohesion) - num(a.cohesion));
    else sorted.sort((a, b) => a.label.localeCompare(b.label));
    return sorted;
  }, [clusters.data, clusterFilter, clusterSort]);

  const targetOptions = (
    <>
      <option value="">Reassign to…</option>
      {clusters.data?.map((cluster) => <option key={cluster.id} value={cluster.id}>{cluster.label}</option>)}
      <option value={NOISE}>Noise</option>
    </>
  );

  if (project.isLoading) return <main className="page">Loading project...</main>;

  return (
    <main className="page project-layout">
      <section className="page-header full">
        <div><h1>{project.data?.name}</h1><p>Status: {project.data?.status}</p></div>
        <div className="header-actions">
          {isOwner && project.data?.status === "ready" && (
            <button className={`button ${editMode ? "primary" : "secondary"}`} onClick={() => setEditMode((prev) => !prev)} type="button">{editMode ? "Done editing" : "Edit clusters"}</button>
          )}
          <Link className="button secondary" to={`/projects/${projectId}/settings`}>Settings</Link>
        </div>
      </section>
      {project.data?.status !== "ready" && <section className="card full"><h2>Pipeline</h2>{status.data?.jobs.map((job) => <div className="job" key={job.step}><span>{job.step}</span><strong>{job.status}</strong></div>)}</section>}
      {project.data?.status === "ready" && <>
        <section className="plot-panel">
          <div className="plot-toolbar">
            {capped && <span className="plot-hint">Overview: showing {displayPoints.length.toLocaleString()} of {points.length.toLocaleString()} points (2D) · small clusters may be under-sampled</span>}
            {capped && editMode && <span className="plot-hint">Lasso-reassign is off on the sampled overview — use the document table to reassign in bulk.</span>}
            {!capped && editMode && mode === "3d" && <span className="plot-hint">Switch to 2D to lasso-select</span>}
            {!capped && <>
              <button className={`button ${mode === "2d" ? "primary" : "secondary"}`} onClick={() => setMode("2d")} type="button">2D</button>
              <button className={`button ${mode === "3d" ? "primary" : "secondary"}`} onClick={() => setMode("3d")} type="button">3D</button>
            </>}
          </div>
          <Plot
            data={[{
              type: webglMode === "3d" ? "scatter3d" : "scattergl",
              mode: "markers",
              x: displayPoints.map((p) => p.x),
              y: displayPoints.map((p) => p.y),
              ...(webglMode === "3d" ? { z: displayPoints.map((p) => p.z ?? 0) } : {}),
              marker: { size: 5, color: colors },
              text: displayPoints.map((p) => hoverHtml(p)),
              hovertemplate: "%{text}<extra></extra>",
            }]}
            layout={{ autosize: true, margin: { l: 0, r: 0, b: 0, t: 0 }, paper_bgcolor: "transparent", plot_bgcolor: "transparent", ...(lassoEnabled ? { dragmode: "lasso" } : {}) }}
            onClick={(event) => {
              const pointNumber = event.points?.[0]?.pointNumber;
              if (pointNumber == null) return;
              const clicked = displayPoints[pointNumber];
              if (!clicked) return;
              setHighlightedClusterId((prev) => (prev === clicked.cluster_id ? null : clicked.cluster_id));
            }}
            onSelected={(event) => {
              if (!editMode) return;
              if (!event || !event.points) { setSelectedPointIds([]); return; }
              const ids = event.points
                .map((p) => displayPoints[(p.pointIndex ?? p.pointNumber) as number]?.document_id)
                .filter((id): id is string => Boolean(id));
              setSelectedPointIds(ids);
            }}
            onDeselect={() => setSelectedPointIds([])}
            useResizeHandler
            className="plot"
          />
          <p className="plot-caveat">UMAP projection — distances and gaps between clusters aren't metric. Use it to spot groups, not to judge how related or far apart they are.</p>
          <div className="plot-legend" role="group" aria-label="Cluster legend">
            {visibleClusters.map((cluster) => {
              const active = highlightedClusterId === cluster.id;
              return (
                <button
                  key={cluster.id}
                  type="button"
                  className={`legend-item ${active ? "active" : ""}`}
                  aria-pressed={active}
                  onClick={() => setHighlightedClusterId((prev) => (prev === cluster.id ? null : cluster.id))}
                >
                  <span className="legend-swatch" style={{ background: clusterColor(clusterIndex.get(cluster.id) ?? 0) }} aria-hidden />
                  {cluster.label}
                </button>
              );
            })}
            {noiseCount > 0 && (
              <span className="legend-item legend-static"><span className="legend-swatch" style={{ background: NOISE_COLOR }} aria-hidden />noise</span>
            )}
          </div>
          {editMode && selectedPointIds.length > 0 && (
            <div className="scatter-selection">
              <span>{selectedPointIds.length} selected</span>
              <select value={reassignTarget} onChange={(event) => setReassignTarget(event.target.value)}>{targetOptions}</select>
              <button className="button" type="button" disabled={!reassignTarget || bulkReassignM.isPending} onClick={() => bulkReassignM.mutate(reassignTarget)}>Reassign</button>
              <input placeholder="New cluster name" value={newClusterLabel} onChange={(event) => setNewClusterLabel(event.target.value)} />
              <button className="button" type="button" disabled={!newClusterLabel.trim() || createFromSelectionM.isPending} onClick={() => createFromSelectionM.mutate(newClusterLabel.trim())}>New cluster</button>
              <button className="button secondary" type="button" onClick={() => setSelectedPointIds([])}>Clear</button>
            </div>
          )}
        </section>
        <section className="cluster-list">
          <div className="cluster-list-controls">
            <input className="cluster-filter" placeholder="Filter clusters…" value={clusterFilter} onChange={(event) => setClusterFilter(event.target.value)} />
            <label className="cluster-sort">Sort
              <select value={clusterSort} onChange={(event) => setClusterSort(event.target.value as typeof clusterSort)}>
                <option value="size">Size</option>
                <option value="sentiment">Sentiment</option>
                <option value="cohesion">Cohesion</option>
                <option value="label">Name</option>
              </select>
            </label>
            <select className="cluster-jump" value="" onChange={(event) => { if (event.target.value) setHighlightedClusterId(event.target.value); }}>
              <option value="">Jump to…</option>
              {visibleClusters.map((cluster) => <option key={cluster.id} value={cluster.id}>{cluster.label}</option>)}
            </select>
          </div>
          {editMode && selectedClusterIds.size >= 2 && (
            <div className="merge-toolbar">
              <span>{selectedClusterIds.size} clusters</span>
              <select value={mergeTarget} onChange={(event) => setMergeTarget(event.target.value)}>
                <option value="">Merge into…</option>
                {[...selectedClusterIds].map((id) => <option key={id} value={id}>{clusters.data?.find((c) => c.id === id)?.label ?? id}</option>)}
              </select>
              <button className="button" type="button" disabled={mergeM.isPending} onClick={handleMergeSelected}>Merge N→1</button>
              <button className="button secondary" type="button" onClick={() => setSelectedClusterIds(new Set())}>Clear</button>
            </div>
          )}
          {visibleClusters.map((cluster) => (
            <ClusterCard
              key={cluster.id}
              cluster={cluster}
              projectId={projectId}
              highlighted={cluster.id === highlightedClusterId}
              cardRef={(node) => { cardRefs.current[cluster.id] = node; }}
              editMode={editMode}
              otherClusters={(clusters.data ?? []).filter((c) => c.id !== cluster.id)}
              selected={selectedClusterIds.has(cluster.id)}
              onToggleSelect={() => toggleClusterSelect(cluster.id)}
              onRename={(label) => renameM.mutate({ id: cluster.id, label, prevLabel: cluster.label })}
              onMergeInto={(targetId) => mergeM.mutate({ sources: [cluster.id], target: targetId })}
              onJunk={() => { if (window.confirm(`Mark "${cluster.label}" as junk? Its documents become noise. This cannot be undone.`)) junkM.mutate(cluster.id); }}
            />
          ))}
          {noiseCount > 0 && (
            <article className="card noise-card">
              <div className="cluster-card-head">
                <span className="noise-swatch" style={{ background: NOISE_COLOR }} aria-hidden />
                <h2>Noise</h2>
              </div>
              <p className="cluster-card-meta">{noiseCount.toLocaleString()} unclustered docs</p>
              <p>Documents left unassigned by clustering. Reassign them in edit mode or from the document table.</p>
            </article>
          )}
        </section>
        <section className="card full">
          <div className="card-row"><h2>All documents</h2><button className="button secondary" type="button" onClick={() => setShowDocuments((prev) => !prev)}>{showDocuments ? "Hide" : "Show all documents"}</button></div>
          {showDocuments && <DocumentsTable projectId={projectId} editable={Boolean(editMode && isOwner)} clusters={clusters.data ?? []} />}
        </section>
        <MetricsPanel projectId={projectId} />
        <EditHistory
          projectId={projectId}
          clusters={clusters.data ?? []}
          isOwner={isOwner}
          onUndo={(edit) => undoM.mutate(edit)}
          undoingId={undoM.isPending ? undoM.variables?.id ?? null : null}
        />
      </>}
    </main>
  );
}

function ClusterCard({ cluster, projectId, highlighted, cardRef, editMode, otherClusters, selected, onToggleSelect, onRename, onMergeInto, onJunk }: {
  cluster: Cluster;
  projectId: string;
  highlighted: boolean;
  cardRef: (node: HTMLElement | null) => void;
  editMode: boolean;
  otherClusters: Cluster[];
  selected: boolean;
  onToggleSelect: () => void;
  onRename: (label: string) => void;
  onMergeInto: (targetId: string) => void;
  onJunk: () => void;
}) {
  const [renaming, setRenaming] = useState(false);
  const [draft, setDraft] = useState(cluster.label);
  // Keep cards a uniform height: collapsed shows one clamped sample + a short word
  // cloud; "Show more" reveals the rest (R15).
  const [expanded, setExpanded] = useState(false);
  const sampleDocs = expanded ? cluster.sample_docs : cluster.sample_docs.slice(0, 1);
  const hasMore = cluster.sample_docs.length > 1 || Object.keys(cluster.word_frequencies ?? {}).length > 10;

  return (
    <article className={`card ${highlighted ? "highlighted" : ""}`} ref={cardRef}>
      <div className="cluster-card-head">
        {editMode && <input type="checkbox" checked={selected} onChange={onToggleSelect} aria-label="Select cluster for merge" />}
        {renaming ? (
          <span className="inline-form">
            <input value={draft} onChange={(event) => setDraft(event.target.value)} autoFocus />
            <button className="button" type="button" disabled={!draft.trim()} onClick={() => { onRename(draft.trim()); setRenaming(false); }}>Save</button>
            <button className="button secondary" type="button" onClick={() => { setDraft(cluster.label); setRenaming(false); }}>Cancel</button>
          </span>
        ) : (
          <h2>{cluster.label}</h2>
        )}
        {!renaming && <LabelSourceBadge labelSource={cluster.label_source} />}
      </div>
      <p>{cluster.summary}</p>
      <p className="cluster-card-meta">{cluster.size} docs · {sentimentSummary(cluster.sentiment_avg, cluster.sentiment_count, cluster.size)}{cluster.mean_stars != null && <StarRating value={cluster.mean_stars} compact />}<CohesionChip value={cluster.cohesion} compact /></p>
      <WordCloud frequencies={cluster.word_frequencies} max={expanded ? 24 : 10} compact />
      {sampleDocs.map((doc) => <blockquote className={expanded ? "" : "clamped"} key={doc.id}>{doc.text}</blockquote>)}
      {hasMore && <button className="card-show-more" type="button" onClick={() => setExpanded((prev) => !prev)}>{expanded ? "Show less" : "Show more"}</button>}
      {editMode ? (
        <div className="actions">
          {!renaming && <button className="button secondary" type="button" onClick={() => { setDraft(cluster.label); setRenaming(true); }}>Rename</button>}
          <select className="cluster-merge-select" value="" onChange={(event) => { if (event.target.value) onMergeInto(event.target.value); }}>
            <option value="">Merge into…</option>
            {otherClusters.map((c) => <option key={c.id} value={c.id}>{c.label}</option>)}
          </select>
          <button className="button secondary danger" type="button" onClick={onJunk}>Mark junk</button>
          <Link className="button" to={`/projects/${projectId}/clusters/${cluster.id}`}>View all</Link>
        </div>
      ) : (
        <Link className="button" to={`/projects/${projectId}/clusters/${cluster.id}`}>View all</Link>
      )}
    </article>
  );
}
