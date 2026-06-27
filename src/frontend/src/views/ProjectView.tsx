import { useQuery } from "@tanstack/react-query";
import Plot from "react-plotly.js";
import { Link, useParams } from "react-router-dom";

import { api } from "../api";

export function ProjectView() {
  const { projectId = "" } = useParams();
  const project = useQuery({ queryKey: ["project", projectId], queryFn: () => api.project(projectId) });
  const status = useQuery({ queryKey: ["pipeline", projectId], queryFn: () => api.pipelineStatus(projectId), refetchInterval: project.data?.status === "ready" ? false : 3000 });
  const embeddings = useQuery({ queryKey: ["embeddings", projectId], queryFn: () => api.embeddings(projectId), enabled: project.data?.status === "ready" });
  const clusters = useQuery({ queryKey: ["clusters", projectId], queryFn: () => api.clusters(projectId), enabled: project.data?.status === "ready" });

  if (project.isLoading) return <main className="page">Loading project...</main>;

  return (
    <main className="page project-layout">
      <section className="page-header full"><div><h1>{project.data?.name}</h1><p>Status: {project.data?.status}</p></div><Link className="button secondary" to={`/projects/${projectId}/settings`}>Settings</Link></section>
      {project.data?.status !== "ready" && <section className="card full"><h2>Pipeline</h2>{status.data?.jobs.map((job) => <div className="job" key={job.step}><span>{job.step}</span><strong>{job.status}</strong></div>)}</section>}
      {project.data?.status === "ready" && <>
        <section className="plot-panel"><Plot data={[{ type: "scatter3d", mode: "markers", x: embeddings.data?.map((p) => p.x), y: embeddings.data?.map((p) => p.y), z: embeddings.data?.map((p) => p.z ?? 0), marker: { size: 5, color: embeddings.data?.map((p) => clusters.data?.findIndex((c) => c.id === p.cluster_id) ?? 0), colorscale: "Viridis" }, text: embeddings.data?.map((p) => p.document_id) }]} layout={{ autosize: true, margin: { l: 0, r: 0, b: 0, t: 0 }, paper_bgcolor: "transparent", plot_bgcolor: "transparent" }} useResizeHandler className="plot" /></section>
        <section className="cluster-list">{clusters.data?.map((cluster) => <article className="card" key={cluster.id}><h2>{cluster.label}</h2><p>{cluster.summary}</p><p>{cluster.size} docs · sentiment {cluster.sentiment_avg ?? "n/a"}</p><div className="terms">{cluster.top_terms.slice(0, 6).map((term) => <span key={term.term}>{term.term}</span>)}</div>{cluster.sample_docs.map((doc) => <blockquote key={doc.id}>{doc.text}</blockquote>)}<Link className="button" to={`/projects/${projectId}/clusters/${cluster.id}`}>View all</Link></article>)}</section>
      </>}
    </main>
  );
}
