import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { FormEvent, useState } from "react";
import { Link } from "react-router-dom";

import { api, Project } from "../api";

export function Dashboard() {
  const queryClient = useQueryClient();
  const projects = useQuery({ queryKey: ["projects"], queryFn: api.projects, refetchInterval: (query) => query.state.data?.some((p) => p.status === "processing" || p.status === "uploading") ? 3000 : false });
  const [showUpload, setShowUpload] = useState(false);
  const upload = useMutation({ mutationFn: api.uploadProject, onSuccess: () => { setShowUpload(false); queryClient.invalidateQueries({ queryKey: ["projects"] }); } });

  return (
    <main className="page dashboard-page">
      <section className="page-header">
        <div><h1>Projects</h1><p>Upload CSV or JSONL files and inspect simulated analysis results.</p></div>
        <button className="primary" onClick={() => setShowUpload(true)}>New Project</button>
      </section>
      {projects.isLoading && <p>Loading projects...</p>}
      {projects.error && <div className="error">{projects.error.message}</div>}
      <section className="project-grid">
        {projects.data?.map((project) => <ProjectCard key={project.id} project={project} />)}
      </section>
      {showUpload && <UploadModal onClose={() => setShowUpload(false)} onSubmit={(form) => upload.mutate(form)} error={upload.error?.message} />}
    </main>
  );
}

function ProjectCard({ project }: { project: Project }) {
  const pipeline = useQuery({
    queryKey: ["pipeline", project.id],
    queryFn: () => api.pipelineStatus(project.id),
    enabled: project.status === "processing" || project.status === "uploading",
    refetchInterval: project.status === "processing" || project.status === "uploading" ? 3000 : false,
  });
  const completedSteps = pipeline.data?.jobs.filter((job) => job.status === "done").length ?? 0;
  const totalSteps = pipeline.data?.jobs.length ?? 8;
  const runningStep = pipeline.data?.jobs.find((job) => job.status === "running")?.step;
  const progress = totalSteps ? Math.round((completedSteps / totalSteps) * 100) : 0;

  return (
    <article className="card project-card">
      <div className="project-card-header">
        <h2>{project.name}</h2>
        <span className={`project-status ${project.status}`}>{project.status}</span>
      </div>
      <div className="project-meta-row">
        <span>{project.role}</span>
        <span>{project.doc_count.toLocaleString()} docs</span>
        <span>{formatDate(project.created_at)}</span>
      </div>
      {project.status === "ready" && <p className="project-card-note">Analysis complete. Clusters and document projections are ready for review.</p>}
      {(project.status === "processing" || project.status === "uploading") && (
        <div className="project-progress">
          <div className="project-progress-label">
            <span>{runningStep ? `Running ${runningStep}` : "Preparing pipeline"}</span>
            <strong>{completedSteps}/{totalSteps}</strong>
          </div>
          <div className="project-progress-track"><span style={{ width: `${progress}%` }} /></div>
        </div>
      )}
      {project.last_error && <div className="error small">{project.last_error}</div>}
      <div className="project-card-divider" />
      <div className="project-card-actions">
        {project.status === "ready" ? (
          <Link className="project-primary-action" to={`/projects/${project.id}`}>Open</Link>
        ) : (
          <Link className="project-primary-action disabled" to={`/projects/${project.id}/pipeline`}>{project.status === "failed" ? "View Error" : "Processing..."}</Link>
        )}
        <div className="project-secondary-actions">
          <Link to={`/projects/${project.id}/pipeline`}>Process</Link>
          <Link to={`/projects/${project.id}/settings`}>Settings</Link>
        </div>
      </div>
    </article>
  );
}

function formatDate(value: string) {
  return new Intl.DateTimeFormat(undefined, { year: "numeric", month: "2-digit", day: "2-digit" }).format(new Date(value));
}

function UploadModal({ onClose, onSubmit, error }: { onClose: () => void; onSubmit: (form: FormData) => void; error?: string }) {
  const [name, setName] = useState("");
  const [file, setFile] = useState<File | null>(null);
  const [columns, setColumns] = useState<{ name: string; type: string; is_primary_key: boolean }[]>([]);

  async function detectColumns(nextFile: File) {
    setFile(nextFile);
    const text = await nextFile.text();
    const names = nextFile.name.endsWith(".jsonl") ? detectJsonlColumns(text) : detectCsvColumns(text);
    setColumns(names.filter(Boolean).map((column, index) => ({ name: column, type: index === 0 ? "text" : "text", is_primary_key: column.toLowerCase() === "id" || index === 0 })));
  }

  function submit(event: FormEvent) {
    event.preventDefault();
    if (!file) return;
    const form = new FormData();
    form.append("name", name);
    form.append("schema_json", JSON.stringify(columns));
    form.append("file", file);
    onSubmit(form);
  }

  return (
    <div className="modal-backdrop">
      <form className="modal" onSubmit={submit}>
        <h2>New Project</h2>
        <label>Project name<input value={name} onChange={(event) => setName(event.target.value)} required /></label>
        <label>Dataset<input type="file" accept=".csv,.jsonl" onChange={(event) => event.target.files?.[0] && detectColumns(event.target.files[0])} required /></label>
        {columns.length > 0 && <div className="schema-box"><h3>Detected schema</h3>{columns.map((column, index) => <div className="schema-row" key={column.name}><span>{column.name}</span><select value={column.type} onChange={(event) => setColumns(columns.map((item, i) => i === index ? { ...item, type: event.target.value } : item))}><option>text</option><option>integer</option><option>float</option><option>date</option><option>boolean</option></select>{column.is_primary_key && <span className="badge">PK</span>}</div>)}</div>}
        {error && <div className="error">{error}</div>}
        <div className="actions"><button type="button" onClick={onClose}>Cancel</button><button className="primary" type="submit">Upload & Start</button></div>
      </form>
    </div>
  );
}

function detectJsonlColumns(text: string) {
  for (const line of text.split(/\r?\n/)) {
    const trimmed = line.trim();
    if (!trimmed) continue;
    try {
      const row = JSON.parse(trimmed) as unknown;
      if (row && typeof row === "object" && !Array.isArray(row)) return Object.keys(row);
    } catch {
      break;
    }
  }
  return ["id", "text"];
}

function detectCsvColumns(text: string) {
  const firstLine = text.split(/\r?\n/).find((line) => line.trim()) ?? "";
  const columns: string[] = [];
  let current = "";
  let inQuotes = false;
  for (let index = 0; index < firstLine.length; index += 1) {
    const char = firstLine[index];
    const next = firstLine[index + 1];
    if (char === '"' && next === '"') {
      current += '"';
      index += 1;
    } else if (char === '"') {
      inQuotes = !inQuotes;
    } else if (char === "," && !inQuotes) {
      columns.push(current.trim());
      current = "";
    } else {
      current += char;
    }
  }
  columns.push(current.trim());
  return columns.length ? columns : ["id", "text"];
}
