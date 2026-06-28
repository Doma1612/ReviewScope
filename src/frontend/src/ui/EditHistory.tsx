import { useQuery } from "@tanstack/react-query";
import { useState } from "react";

import { api, Cluster, ClusterEdit } from "../api";

// Surface the cluster-edit audit log (GET /edits) as a collapsible
// history panel, newest-first, with an Undo button on the reversible actions.

// Undo is feasible only where the edit records enough "before" state to invert it:
// single reassign stores the doc's old cluster, bulk reassign stores a per-doc
// `before` map, and rename stores the previous label. Merges/junk/creates are
// destructive (membership is fanned out) and stay non-undoable for v1.
export function isUndoable(edit: ClusterEdit): boolean {
  switch (edit.action) {
    case "reassign_doc":
      return Boolean(edit.document_id);
    case "bulk_reassign":
      return edit.payload != null && typeof edit.payload.before === "object" && edit.payload.before !== null;
    case "rename_label":
      return edit.payload != null && typeof edit.payload.before === "string";
    default:
      return false;
  }
}

const NOISE_LABEL = "Noise";

function describe(edit: ClusterEdit, clusterLabel: (id: string | null) => string): string {
  const target = () => clusterLabel(edit.target_cluster_id);
  switch (edit.action) {
    case "reassign_doc":
      return `Moved a document → ${target()}`;
    case "bulk_reassign": {
      const count = Array.isArray(edit.payload?.document_ids) ? (edit.payload!.document_ids as unknown[]).length : 0;
      return `Moved ${count} document${count === 1 ? "" : "s"} → ${target()}`;
    }
    case "merge_clusters":
      return `Merged a cluster → ${target()}`;
    case "create_cluster":
      return `Created cluster “${edit.new_label ?? ""}”`;
    case "create_from_selection":
      return `Created “${edit.new_label ?? ""}” from a selection`;
    case "rename_label":
      return `Renamed a cluster to “${edit.new_label ?? ""}”`;
    case "approve_label":
      return "Approved a cluster label";
    case "mark_junk":
      return "Marked a cluster as junk";
    case "split_cluster":
      return "Split a cluster";
    case "confirm_run":
      return "Confirmed the run";
    default:
      return edit.action;
  }
}

export function EditHistory({
  projectId,
  clusters,
  isOwner,
  onUndo,
  undoingId,
}: {
  projectId: string;
  clusters: Cluster[];
  isOwner: boolean;
  onUndo: (edit: ClusterEdit) => void;
  undoingId: string | null;
}) {
  const [open, setOpen] = useState(false);
  const edits = useQuery({ queryKey: ["edits", projectId], queryFn: () => api.edits(projectId), enabled: open });
  const members = useQuery({ queryKey: ["members", projectId], queryFn: () => api.members(projectId), enabled: open });

  // Cluster ids are regenerated on re-run and deleted clusters drop out of the
  // list, so resolution falls back to the edit's stored label, then a short id.
  const labelById = new Map(clusters.map((c) => [c.id, c.label]));
  const clusterLabel = (id: string | null): string => (id == null ? NOISE_LABEL : labelById.get(id) ?? `cluster ${id.slice(0, 8)}`);
  const actorById = new Map((members.data ?? []).map((m) => [m.user_id, m.email]));

  return (
    <section className="card full edit-history">
      <div className="card-row">
        <h2>Edit history</h2>
        <button className="button secondary" type="button" onClick={() => setOpen((prev) => !prev)}>{open ? "Hide" : "Show edit history"}</button>
      </div>
      {open && (
        edits.isLoading ? <p className="muted">Loading…</p>
        : (edits.data?.length ?? 0) === 0 ? <p className="muted">No edits yet.</p>
        : <ul className="edit-list">
            {edits.data!.map((edit) => {
              const undoable = isOwner && isUndoable(edit);
              return (
                <li className="edit-row" key={edit.id}>
                  <div className="edit-main">
                    <span className="edit-desc">{describe(edit, clusterLabel)}</span>
                    <span className="edit-meta">{actorById.get(edit.actor_id) ?? "Unknown"} · {new Date(edit.created_at).toLocaleString()}</span>
                  </div>
                  {undoable && (
                    <button className="button secondary" type="button" disabled={undoingId === edit.id} onClick={() => onUndo(edit)}>
                      {undoingId === edit.id ? "Undoing…" : "Undo"}
                    </button>
                  )}
                </li>
              );
            })}
          </ul>
      )}
    </section>
  );
}
