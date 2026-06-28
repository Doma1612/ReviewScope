import { useQuery } from "@tanstack/react-query";
import { useEffect, useState } from "react";
import { Link } from "react-router-dom";

import { api, Cluster, EmbeddingPoint, Project } from "../api";
import { hoverTitle } from "../hover";
import { samplePoints } from "../plot";

const PALETTE = ["#38bdf8", "#a78bfa", "#34d399", "#f59e0b", "#fb7185", "#22d3ee", "#f472b6", "#bef264"];

export function DeckDashboard() {
  const [selectedProjectId, setSelectedProjectId] = useState<string | null>(null);
  const [highlightedClusterId, setHighlightedClusterId] = useState<string | null>(null);
  const projects = useQuery({ queryKey: ["projects"], queryFn: api.projects, refetchInterval: (query) => query.state.data?.some((project) => project.status !== "ready" && project.status !== "failed") ? 3000 : false });
  const selectedProject = projects.data?.find((project) => project.id === selectedProjectId) ?? projects.data?.find((project) => project.status === "ready") ?? projects.data?.[0];
  const embeddings = useQuery({ queryKey: ["deck-embeddings", selectedProject?.id], queryFn: () => api.embeddings(selectedProject!.id), enabled: selectedProject?.status === "ready" });
  const clusters = useQuery({ queryKey: ["deck-clusters", selectedProject?.id], queryFn: () => api.clusters(selectedProject!.id), enabled: selectedProject?.status === "ready" });

  useEffect(() => {
    if (!selectedProjectId && projects.data?.length) {
      setSelectedProjectId(projects.data.find((project) => project.status === "ready")?.id ?? projects.data[0].id);
    }
  }, [projects.data, selectedProjectId]);

  useEffect(() => {
    setHighlightedClusterId(null);
  }, [selectedProject?.id]);

  const readyProjects = projects.data?.filter((project) => project.status === "ready").length ?? 0;
  const totalDocs = projects.data?.reduce((sum, project) => sum + project.doc_count, 0) ?? 0;
  const clusterLookup = new Map((clusters.data ?? []).map((cluster, index) => [cluster.id, { cluster, color: PALETTE[index % PALETTE.length] }]));

  return (
    <main className="deck-page">
      <section className="deck-hero">
        <div>
          <p className="deck-kicker">Spatial Analysis Console</p>
          <h1>Cluster Atlas</h1>
          <p>Deck.gl-inspired point-cloud dashboard for scanning project topology before opening the detailed Plotly view.</p>
        </div>
        <div className="deck-metrics">
          <Metric label="Projects" value={projects.data?.length ?? 0} />
          <Metric label="Ready" value={readyProjects} />
          <Metric label="Documents" value={totalDocs} />
        </div>
      </section>

      <section className="deck-grid">
        <aside className="deck-panel deck-projects">
          <div className="deck-panel-title">
            <span>Project Stream</span>
            <Link to="/" className="deck-link">classic upload</Link>
          </div>
          {projects.data?.map((project) => (
            <button
              className={`deck-project ${project.id === selectedProject?.id ? "active" : ""}`}
              key={project.id}
              onClick={() => setSelectedProjectId(project.id)}
              type="button"
            >
              <span>
                <strong>{project.name}</strong>
                <small>{project.doc_count} docs · {project.role}</small>
              </span>
              <em className={`deck-status ${project.status}`}>{project.status}</em>
            </button>
          ))}
          {!projects.data?.length && <p className="deck-muted">No projects yet. Upload a dataset from the classic dashboard.</p>}
        </aside>

        <section className="deck-map-card">
          <div className="deck-map-header">
            <div>
              <p className="deck-kicker">{selectedProject?.status ?? "No Project"}</p>
              <h2>{selectedProject?.name ?? "Select a Project"}</h2>
            </div>
            {selectedProject?.status === "ready" && <Link className="deck-open" to={`/projects/${selectedProject.id}`}>Open Detail</Link>}
          </div>
          <PointCloud points={embeddings.data ?? []} clusters={clusters.data ?? []} highlightedClusterId={highlightedClusterId} />
          {selectedProject && selectedProject.status !== "ready" && (
            <div className="deck-map-empty">
              <strong>{selectedProject.status}</strong>
              <span>The spatial canvas unlocks when the simulated pipeline marks this project ready.</span>
            </div>
          )}
        </section>

        <aside className="deck-panel deck-clusters">
          <div className="deck-panel-title">
            <span>Cluster Layers</span>
            <span>{clusters.data?.length ?? 0}</span>
          </div>
          {clusters.data?.map((cluster) => (
            <ClusterLayer
              cluster={cluster}
              color={clusterLookup.get(cluster.id)?.color ?? PALETTE[0]}
              key={cluster.id}
              onHover={setHighlightedClusterId}
              selectedProjectId={selectedProject?.id}
            />
          ))}
          {selectedProject?.status === "ready" && clusters.isLoading && <p className="deck-muted">Loading layers...</p>}
        </aside>
      </section>
    </main>
  );
}

function Metric({ label, value }: { label: string; value: number }) {
  return (
    <div className="deck-metric">
      <strong>{value.toLocaleString()}</strong>
      <span>{label}</span>
    </div>
  );
}

function PointCloud({ points, clusters, highlightedClusterId }: { points: EmbeddingPoint[]; clusters: Cluster[]; highlightedClusterId: string | null }) {
  // Cap the rendered DOM nodes (F6) — bounds come from the full set so the layout
  // stays stable, but we only paint a sampled subset for huge projects.
  const bounds = getBounds(points);
  const displayPoints = samplePoints(points);
  const capped = displayPoints.length < points.length;
  const clusterIndex = new Map(clusters.map((cluster, index) => [cluster.id, index]));

  return (
    <div className="deck-map">
      <div className="deck-gridlines" />
      <div className="deck-vignette" />
      {displayPoints.map((point) => {
        const index = point.cluster_id ? clusterIndex.get(point.cluster_id) ?? 0 : 0;
        const color = PALETTE[index % PALETTE.length];
        const muted = highlightedClusterId && point.cluster_id !== highlightedClusterId;
        return (
          <span
            className={`deck-point ${muted ? "muted" : ""}`}
            key={point.document_id}
            style={{
              "--point-color": color,
              left: `${scale(point.x, bounds.minX, bounds.maxX)}%`,
              top: `${100 - scale(point.y, bounds.minY, bounds.maxY)}%`,
            } as React.CSSProperties}
            title={hoverTitle(point)}
          />
        );
      })}
      <div className="deck-map-caption">
        <span>{capped ? `Showing ${displayPoints.length.toLocaleString()} of ${points.length.toLocaleString()}` : `${points.length.toLocaleString()} projected documents`}</span>
        <span>UMAP x/y · simulated layer data</span>
      </div>
    </div>
  );
}

function ClusterLayer({ cluster, color, onHover, selectedProjectId }: { cluster: Cluster; color: string; onHover: (clusterId: string | null) => void; selectedProjectId?: string }) {
  return (
    <Link
      className="deck-layer"
      onMouseEnter={() => onHover(cluster.id)}
      onMouseLeave={() => onHover(null)}
      style={{ "--layer-color": color } as React.CSSProperties}
      to={selectedProjectId ? `/projects/${selectedProjectId}/clusters/${cluster.id}` : "#"}
    >
      <span className="deck-layer-swatch" />
      <span>
        <strong>{cluster.label}</strong>
        <small>{cluster.size} docs · sentiment {cluster.sentiment_avg ?? "n/a"}</small>
      </span>
    </Link>
  );
}

function getBounds(points: EmbeddingPoint[]) {
  const xs = points.map((point) => point.x);
  const ys = points.map((point) => point.y);
  return {
    minX: Math.min(...xs, -1),
    maxX: Math.max(...xs, 1),
    minY: Math.min(...ys, -1),
    maxY: Math.max(...ys, 1),
  };
}

function scale(value: number, min: number, max: number) {
  if (max === min) return 50;
  return Math.min(96, Math.max(4, ((value - min) / (max - min)) * 92 + 4));
}
