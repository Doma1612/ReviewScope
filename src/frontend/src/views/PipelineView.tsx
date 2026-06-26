import { useQuery } from "@tanstack/react-query";
import { Link, useParams } from "react-router-dom";

import { api, PipelineStatus } from "../api";

type MethodStage = {
  key: string;
  title: string;
  methodology: string;
  output: string;
  risk: string;
  next: string;
};

const METHOD_STAGES: MethodStage[] = [
  {
    key: "upload",
    title: "Upload & Schema Confirmation",
    methodology: "The application accepts a structured CSV, JSON, or JSONL corpus and records the user-confirmed schema before analysis starts.",
    output: "A project shell, stored source file, schema columns, and queued pipeline jobs.",
    risk: "The empirical model defaults were tuned on hotel review text, so arbitrary uploads may behave differently.",
    next: "Validate rows and identify the document text used for analysis.",
  },
  {
    key: "ingest",
    title: "Data Ingestion & Sampling",
    methodology: "The methodology starts from a benchmark corpus of text documents, with data filtered into a comparable working set.",
    output: "Readable rows with primary keys, raw metadata, and candidate text fields.",
    risk: "Sampling and filtering can bias what the model sees; malformed rows may be skipped in this simulated service.",
    next: "Clean text minimally and remove invalid or duplicate documents.",
  },
  {
    key: "preprocess",
    title: "Minimal Preprocessing",
    methodology: "Whitespace normalization is preferred; aggressive lowercasing or punctuation stripping hurt downstream cluster quality in the benchmark.",
    output: "Normalized document text, deduplicated records, and short-text filtering.",
    risk: "Near-duplicates and very short texts remain a known weakness in real corpora.",
    next: "Convert each document into a semantic embedding vector.",
  },
  {
    key: "embed",
    title: "Embedding Generation",
    methodology: "The selected benchmark model is all-mpnet-base-v2 without instruction prompts; embeddings represent semantic similarity.",
    output: "One vector representation per document for similarity and clustering.",
    risk: "The selected default is English benchmark-driven; multilingual or domain-shifted corpora need re-evaluation.",
    next: "Reduce the high-dimensional vectors into a clusterable low-dimensional space.",
  },
  {
    key: "reduce",
    title: "Dimensionality Reduction",
    methodology: "UMAP reduces embeddings for clustering and separate 2D/3D visualization projections are produced for exploration.",
    output: "UMAP coordinates for visualization and lower-dimensional structure for cluster assignment.",
    risk: "UMAP can distort distances and densities; visual separation is not proof of semantic truth.",
    next: "Group nearby documents into candidate themes.",
  },
  {
    key: "cluster",
    title: "Clustering",
    methodology: "The evaluated candidates include HDBSCAN, agglomerative clustering, BERTopic baseline, and a two-stage micro-to-macro clusterer.",
    output: "Cluster IDs, cluster sizes, and noise or uncertain assignments.",
    risk: "Algorithms may discard hard documents as noise or create clusters from projection artifacts.",
    next: "Check whether clusters are sentiment blobs or real themes.",
  },
  {
    key: "sentiment",
    title: "Sentiment & Rating Check",
    methodology: "Tier 3 exists because embeddings can cluster by sentiment instead of topic; mixed sentiment within a cluster suggests a real theme.",
    output: "Per-document sentiment scores and cluster-level averages or distributions.",
    risk: "A polarized real topic can look like a sentiment blob, so human inspection still matters.",
    next: "Extract representative terms and produce human-readable labels.",
  },
  {
    key: "label",
    title: "Representation & LLM Labeling",
    methodology: "c-TF-IDF terms and representative documents feed the label/summary stage; labels require human review because hallucination is possible.",
    output: "Cluster labels, summaries, top terms, word frequencies, and sample documents.",
    risk: "Labels may overfit sampled examples or sound plausible while being wrong.",
    next: "Persist artifacts and expose the project for exploration.",
  },
  {
    key: "finalize",
    title: "Finalize & Review Readiness",
    methodology: "The run becomes reviewable only after artifacts are persisted. The methodology states humans must remain in the loop for final decisions.",
    output: "Ready project, visualization coordinates, clusters, documents, and status history.",
    risk: "Automated metrics and labels are proxies; final interpretation needs analyst confirmation.",
    next: "Open the cluster view and inspect labels, samples, terms, and documents.",
  },
];

const STATUS_ORDER = { pending: 0, running: 1, done: 2, failed: 3 } as const;

export function PipelineView() {
  const { projectId = "" } = useParams();
  const project = useQuery({ queryKey: ["project", projectId], queryFn: () => api.project(projectId) });
  const status = useQuery({
    queryKey: ["pipeline", projectId],
    queryFn: () => api.pipelineStatus(projectId),
    refetchInterval: (query) => {
      const state = query.state.data?.status;
      return state === "ready" || state === "failed" ? false : 2000;
    },
  });

  const stageStates = getStageStates(status.data);
  const currentStage = stageStates.find((stage) => stage.state === "running") ?? stageStates.find((stage) => stage.state === "failed") ?? stageStates.find((stage) => stage.state === "pending") ?? stageStates.at(-1);
  const completedCount = stageStates.filter((stage) => stage.state === "done").length;
  const progress = Math.round((completedCount / stageStates.length) * 100);

  return (
    <main className="page pipeline-page">
      <section className="page-header pipeline-header">
        <div>
          <p className="pipeline-eyebrow">Model Training Process</p>
          <h1>{project.data?.name ?? "Pipeline"}</h1>
          <p>Follow the model workflow from upload to review-ready clusters, based on the methodology document.</p>
        </div>
        <div className="actions">
          <Link className="button secondary" to="/">Dashboard</Link>
          {project.data?.status === "ready" && <Link className="button primary" to={`/projects/${projectId}`}>Open Results</Link>}
        </div>
      </section>

      <section className="pipeline-overview card">
        <div>
          <span className={`badge ${project.data?.status ?? "processing"}`}>{project.data?.status ?? "loading"}</span>
          <h2>{currentStage?.title ?? "Preparing pipeline"}</h2>
          <p>{currentStage?.methodology}</p>
        </div>
        <div className="pipeline-progress-ring" style={{ "--progress": `${progress}%` } as React.CSSProperties}>
          <strong>{progress}%</strong>
          <span>complete</span>
        </div>
      </section>

      {project.data?.last_error && <div className="error">{project.data.last_error}</div>}

      <section className="pipeline-layout">
        <ol className="pipeline-timeline">
          {stageStates.map((stage, index) => (
            <li className={`pipeline-step ${stage.state}`} key={stage.key}>
              <div className="pipeline-step-marker">{index + 1}</div>
              <div className="pipeline-step-body">
                <div className="pipeline-step-title">
                  <h2>{stage.title}</h2>
                  <span>{stage.state}</span>
                </div>
                <p>{stage.methodology}</p>
                <div className="pipeline-step-meta">
                  <strong>Result:</strong> {getResultText(stage)}
                </div>
                {stage.message && <div className="pipeline-step-message">{stage.message}</div>}
              </div>
            </li>
          ))}
        </ol>

        <aside className="pipeline-inspector">
          <article className="card">
            <p className="pipeline-eyebrow">Current Stage</p>
            <h2>{currentStage?.title}</h2>
            <p>{currentStage?.methodology}</p>
            <dl>
              <dt>Expected result</dt>
              <dd>{currentStage?.output}</dd>
              <dt>Known risk</dt>
              <dd>{currentStage?.risk}</dd>
              <dt>What comes next</dt>
              <dd>{currentStage?.next}</dd>
            </dl>
          </article>

          <article className="card">
            <p className="pipeline-eyebrow">Methodology Notes</p>
            <ul className="pipeline-notes">
              <li>Metrics are proxies; humans decide final cluster quality.</li>
              <li>UMAP visual separation is helpful but can distort density.</li>
              <li>LLM labels are review candidates, not ground truth.</li>
              <li>Sentiment checks help detect rating-based clusters.</li>
            </ul>
          </article>
        </aside>
      </section>
    </main>
  );
}

function getStageStates(status?: PipelineStatus) {
  const jobs = new Map(status?.jobs.map((job) => [job.step, job]) ?? []);
  return METHOD_STAGES.map((stage) => {
    if (stage.key === "upload") {
      return { ...stage, state: "done", message: "File accepted and project created." };
    }
    const job = jobs.get(stage.key);
    return {
      ...stage,
      state: job?.status ?? "pending",
      message: job?.message ?? null,
    };
  }).sort((a, b) => METHOD_STAGES.findIndex((stage) => stage.key === a.key) - METHOD_STAGES.findIndex((stage) => stage.key === b.key));
}

function getResultText(stage: MethodStage & { state: string; message: string | null }) {
  if (stage.state === "done") return stage.output;
  if (stage.state === "running") return "In progress. The service is generating this stage's artifacts now.";
  if (stage.state === "failed") return "This stage failed. Review the error and rerun after fixing the input or service issue.";
  return "Waiting for previous stages to complete.";
}
