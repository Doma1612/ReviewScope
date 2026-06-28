import { useQuery } from "@tanstack/react-query";
import { useState } from "react";

import { api } from "../api";

// Friendly labels for the run-level metric keys the ML harness emits. Unknown keys
// fall back to their raw name so new metrics still show up.
const LABELS: Record<string, string> = {
  silhouette: "Silhouette (excl. noise)",
  silhouette_incl_noise: "Silhouette (incl. noise)",
  davies_bouldin: "Davies–Bouldin",
  calinski_harabasz: "Calinski–Harabasz",
  coherence_cv: "Topic coherence (C_v)",
  rating_entropy: "Rating entropy",
  n_clusters: "Clusters",
  noise_fraction: "Noise fraction",
  noise_ratio: "Noise fraction",
};

function formatValue(value: unknown): string | null {
  if (value == null) return null;
  if (typeof value === "boolean") return value ? "yes" : "no";
  if (typeof value === "number") return Number.isInteger(value) ? String(value) : value.toFixed(3);
  if (typeof value === "string") return value;
  return null;
}

// R17 — collapsible "Clustering quality" panel surfacing the run-level report the
// pipeline already computes (silhouette/coherence/rating-entropy/…). Hidden details
// are fetched lazily on open. Real runs only — simulated runs have no real geometry.
export function MetricsPanel({ projectId }: { projectId: string }) {
  const [open, setOpen] = useState(false);
  const metrics = useQuery({ queryKey: ["project-metrics", projectId], queryFn: () => api.projectMetrics(projectId), enabled: open });

  const data = metrics.data;
  const entries = data?.metrics
    ? Object.entries(data.metrics)
        .map(([key, value]) => [LABELS[key] ?? key, formatValue(value)] as const)
        .filter(([, value]) => value !== null)
    : [];

  return (
    <section className="card full metrics-panel">
      <div className="card-row">
        <h2>Clustering quality</h2>
        <button className="button secondary" type="button" onClick={() => setOpen((prev) => !prev)}>{open ? "Hide" : "Show"}</button>
      </div>
      {open && (
        <>
          {metrics.isLoading && <p className="documents-table-empty">Loading metrics…</p>}
          {!metrics.isLoading && !data?.metrics && (
            <p className="documents-table-empty">No quality metrics for this run (only computed for real, non-simulated runs).</p>
          )}
          {data?.metrics && (
            <>
              {data.stale && <p className="metrics-stale">⚠ These figures reflect the original run, not your later edits.</p>}
              <dl className="metrics-grid">
                {entries.map(([label, value]) => (
                  <div className="metrics-item" key={label}>
                    <dt>{label}</dt>
                    <dd>{value}</dd>
                  </div>
                ))}
              </dl>
              <p className="metrics-caveat">
                Read these together: instruction-tuned embeddings can inflate silhouette mechanically without better topics, so weigh it against
                topic coherence. Silhouette (excl. noise) is computed on the easier non-noise remainder.
              </p>
            </>
          )}
        </>
      )}
    </section>
  );
}
