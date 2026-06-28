import { useQuery } from "@tanstack/react-query";
import { useMemo, useState } from "react";
import Plot from "react-plotly.js";
import { Link, useParams } from "react-router-dom";

import { api, DocumentItem } from "../api";
import { sentimentSummary } from "../hover";
import { CohesionChip } from "../ui/CohesionChip";
import { DocumentsTable } from "../ui/DocumentsTable";
import { LabelSourceBadge } from "../ui/LabelSourceBadge";
import { StarRating } from "../ui/StarRating";
import { WordCloud } from "../ui/WordCloud";

// Bucket the cluster's per-document sentiment scores into negative/neutral/positive
// for the distribution chart. Mirrors the thresholds in hover.sentimentLabel.
function sentimentBuckets(documents: DocumentItem[]) {
  const counts = { negative: 0, neutral: 0, positive: 0 };
  for (const doc of documents) {
    if (doc.sentiment_score == null) continue;
    if (doc.sentiment_score > 0.05) counts.positive += 1;
    else if (doc.sentiment_score < -0.05) counts.negative += 1;
    else counts.neutral += 1;
  }
  return counts;
}

export function ClusterDetail() {
  const { projectId = "", clusterId = "" } = useParams();
  const project = useQuery({ queryKey: ["project", projectId], queryFn: () => api.project(projectId) });
  const cluster = useQuery({ queryKey: ["cluster", projectId, clusterId], queryFn: () => api.cluster(projectId, clusterId) });
  const clusters = useQuery({ queryKey: ["clusters", projectId], queryFn: () => api.clusters(projectId) });
  const documents = useQuery({ queryKey: ["cluster-docs", projectId, clusterId], queryFn: () => api.clusterDocuments(projectId, clusterId) });

  const [editMode, setEditMode] = useState(false);
  const isOwner = project.data?.role === "owner";

  const buckets = useMemo(() => sentimentBuckets(documents.data ?? []), [documents.data]);
  const hasSentiment = buckets.negative + buckets.neutral + buckets.positive > 0;

  return (
    <main className="page">
      <Link to={`/projects/${projectId}`}>Back to cluster view</Link>
      <section className="page-header">
        <div><h1 className="cluster-detail-title">{cluster.data?.label ?? "Cluster"}{cluster.data && <LabelSourceBadge labelSource={cluster.data.label_source} />}</h1><p className="cluster-detail-meta">{cluster.data?.size ?? 0} documents · {cluster.data ? sentimentSummary(cluster.data.sentiment_avg, cluster.data.sentiment_count, cluster.data.size) : "sentiment n/a"}{cluster.data?.mean_stars != null && <> · <StarRating value={cluster.data.mean_stars} /></>}{cluster.data?.cohesion != null && <CohesionChip value={cluster.data.cohesion} />}</p></div>
        {isOwner && <button className={`button ${editMode ? "primary" : "secondary"}`} onClick={() => setEditMode((prev) => !prev)} type="button">{editMode ? "Done editing" : "Edit documents"}</button>}
      </section>
      <section className="card"><p>{cluster.data?.summary}</p><WordCloud frequencies={cluster.data?.word_frequencies} /><div className="terms">{cluster.data?.top_terms.map((term) => <span key={term.term}>{term.term} {term.score}</span>)}</div></section>
      {hasSentiment && (
        <section className="card">
          <h2>Sentiment distribution</h2>
          <Plot
            data={[{
              type: "bar",
              x: ["negative", "neutral", "positive"],
              y: [buckets.negative, buckets.neutral, buckets.positive],
              marker: { color: ["#fb7185", "#94a3b8", "#34d399"] },
            }]}
            layout={{ autosize: true, height: 280, margin: { l: 40, r: 10, b: 40, t: 10 }, paper_bgcolor: "transparent", plot_bgcolor: "transparent", yaxis: { title: "documents" } }}
            useResizeHandler
            style={{ width: "100%" }}
          />
        </section>
      )}
      <section className="card"><h2>Documents</h2><DocumentsTable projectId={projectId} clusterId={clusterId} editable={Boolean(editMode && isOwner)} clusters={clusters.data ?? []} /></section>
    </main>
  );
}
