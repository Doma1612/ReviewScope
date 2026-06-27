import { useQuery } from "@tanstack/react-query";
import { Link, useParams } from "react-router-dom";

import { api } from "../api";

export function ClusterDetail() {
  const { projectId = "", clusterId = "" } = useParams();
  const cluster = useQuery({ queryKey: ["cluster", projectId, clusterId], queryFn: () => api.cluster(projectId, clusterId) });
  const documents = useQuery({ queryKey: ["cluster-docs", projectId, clusterId], queryFn: () => api.clusterDocuments(projectId, clusterId) });

  return (
    <main className="page">
      <Link to={`/projects/${projectId}`}>Back to cluster view</Link>
      <section className="page-header"><div><h1>{cluster.data?.label ?? "Cluster"}</h1><p>{cluster.data?.size ?? 0} documents · sentiment {cluster.data?.sentiment_avg ?? "n/a"}</p></div></section>
      <section className="card"><p>{cluster.data?.summary}</p><div className="terms">{cluster.data?.top_terms.map((term) => <span key={term.term}>{term.term} {term.score}</span>)}</div></section>
      <section className="card"><h2>Documents</h2><div className="table">{documents.data?.map((doc) => <article key={doc.id} className="doc-row"><strong>{doc.primary_key_value}</strong><p>{doc.text}</p></article>)}</div></section>
    </main>
  );
}
