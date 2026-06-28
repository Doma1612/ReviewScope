import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { FormEvent, useState } from "react";
import { Link } from "react-router-dom";

import { api, Project } from "../api";
import { useSimulated } from "../useSimulated";

export function Dashboard() {
  const queryClient = useQueryClient();
  const projects = useQuery({ queryKey: ["projects"], queryFn: api.projects, refetchInterval: (query) => query.state.data?.some((p) => p.status === "processing" || p.status === "uploading") ? 3000 : false });
  const [showUpload, setShowUpload] = useState(false);
  const simulated = useSimulated();
  const upload = useMutation({ mutationFn: api.uploadProject, onSuccess: () => { setShowUpload(false); queryClient.invalidateQueries({ queryKey: ["projects"] }); } });

  return (
    <main className="page">
      <section className="page-header">
        <div><h1>Projects</h1><p>Upload CSV or JSONL files and inspect {simulated ? "simulated " : ""}analysis results.</p></div>
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
      {project.role === "viewer" && project.owner_email && <p className="project-card-owner">Shared by {project.owner_email}</p>}
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

type SchemaCol = { name: string; type: string; is_primary_key: boolean };
const COLUMN_TYPES = ["text", "integer", "float", "date", "boolean"];

function UploadModal({ onClose, onSubmit, error }: { onClose: () => void; onSubmit: (form: FormData) => void; error?: string }) {
  const [name, setName] = useState("");
  const [file, setFile] = useState<File | null>(null);
  const [columns, setColumns] = useState<SchemaCol[]>([]);
  const [errors, setErrors] = useState<string[]>([]);

  async function detectColumns(nextFile: File) {
    setFile(nextFile);
    setErrors([]);
    const text = await nextFile.text();
    setColumns(buildColumns(text, nextFile.name.endsWith(".jsonl")));
  }

  // Validation is submit-time only (no auto-fixing): editing a field clears the
  // stale error banner so the user re-validates on their next submit.
  const setType = (index: number, type: string) => { setErrors([]); setColumns((cs) => cs.map((c, i) => (i === index ? { ...c, type } : c))); };
  const setPrimaryKey = (index: number) => { setErrors([]); setColumns((cs) => cs.map((c, i) => ({ ...c, is_primary_key: i === index }))); };

  function submit(event: FormEvent) {
    event.preventDefault();
    if (!file) { setErrors(["Choose a dataset file."]); return; }
    const problems = validateSchema(columns);
    if (problems.length) { setErrors(problems); return; }
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
        {columns.length > 0 && (
          <div className="schema-box">
            <h3>Confirm schema</h3>
            <p className="schema-hint">Pick the primary key (one row) and the column type for each field. The text column holds the content to analyze.</p>
            {columns.map((column, index) => (
              <div className="schema-row" key={column.name}>
                <span className="schema-col-name">{column.name}</span>
                <select value={column.type} onChange={(event) => setType(index, event.target.value)}>
                  {COLUMN_TYPES.map((type) => <option key={type} value={type}>{type}</option>)}
                </select>
                <label className="schema-pk"><input type="radio" name="primary-key" checked={column.is_primary_key} onChange={() => setPrimaryKey(index)} />PK</label>
              </div>
            ))}
          </div>
        )}
        {errors.map((message) => <div className="error small" key={message}>{message}</div>)}
        {error && <div className="error">{error}</div>}
        <div className="actions"><button type="button" onClick={onClose}>Cancel</button><button className="primary" type="submit">Upload & Start</button></div>
      </form>
    </div>
  );
}

// ── Schema detection ────────────────────────────────────────────────────────────
// Sample the head of the file and infer each column's type + the primary key from
// real values, rather than defaulting everything to text / the first column.
const SAMPLE_ROWS = 50;
const BOOL_VALUES = new Set(["true", "false", "yes", "no"]);
const ID_NAME_RE = /(^|[_\s-])(id|key|uuid|pk)$/i;
const DATE_RE = /^\d{4}-\d{1,2}-\d{1,2}([t ].*)?$/i;

function parseCsvLine(line: string): string[] {
  const cells: string[] = [];
  let current = "";
  let inQuotes = false;
  for (let index = 0; index < line.length; index += 1) {
    const char = line[index];
    const next = line[index + 1];
    if (char === '"' && next === '"') { current += '"'; index += 1; }
    else if (char === '"') { inQuotes = !inQuotes; }
    else if (char === "," && !inQuotes) { cells.push(current.trim()); current = ""; }
    else { current += char; }
  }
  cells.push(current.trim());
  return cells;
}

// Up to SAMPLE_ROWS non-empty values per column, for CSV or JSONL.
function sampleColumns(text: string, isJsonl: boolean): { name: string; values: string[] }[] {
  const clean = (values: unknown[]) => values.filter((v) => v != null && v !== "").map(String);
  if (isJsonl) {
    const objects: Record<string, unknown>[] = [];
    for (const line of text.split(/\r?\n/)) {
      const trimmed = line.trim();
      if (!trimmed) continue;
      try {
        const row = JSON.parse(trimmed) as unknown;
        if (row && typeof row === "object" && !Array.isArray(row)) objects.push(row as Record<string, unknown>);
      } catch { break; }
      if (objects.length >= SAMPLE_ROWS) break;
    }
    if (!objects.length) return [{ name: "id", values: [] }, { name: "text", values: [] }];
    return Object.keys(objects[0]).map((name) => ({ name, values: clean(objects.map((o) => o[name])) }));
  }
  const lines = text.split(/\r?\n/).filter((line) => line.trim());
  if (!lines.length) return [{ name: "id", values: [] }, { name: "text", values: [] }];
  const names = parseCsvLine(lines[0]);
  const rows = lines.slice(1, SAMPLE_ROWS + 1).map(parseCsvLine);
  return names.map((name, col) => ({ name, values: clean(rows.map((r) => r[col])) }));
}

function inferType(values: string[]): string {
  if (!values.length) return "text";
  const all = (pred: (value: string) => boolean) => values.every(pred);
  if (all((v) => BOOL_VALUES.has(v.toLowerCase()))) return "boolean";
  if (all((v) => /^-?\d+$/.test(v))) return "integer";
  if (values.some((v) => v.includes(".")) && all((v) => /^-?(\d+\.?\d*|\.\d+)$/.test(v))) return "float";
  if (all((v) => DATE_RE.test(v) && !Number.isNaN(Date.parse(v)))) return "date";
  return "text";
}

// Prefer a column with unique, non-empty values and an id-like name; otherwise the
// first all-unique column; otherwise an id-named column; otherwise the first column.
function detectPrimaryKey(cols: { name: string; values: string[] }[]): number {
  const unique = (values: string[]) => values.length > 0 && new Set(values).size === values.length;
  const candidates = cols.map((c, i) => ({ i, c })).filter(({ c }) => unique(c.values));
  const named = candidates.find(({ c }) => ID_NAME_RE.test(c.name) || c.name.toLowerCase() === "id");
  if (named) return named.i;
  if (candidates.length) return candidates[0].i;
  const byName = cols.findIndex((c) => ID_NAME_RE.test(c.name) || c.name.toLowerCase() === "id");
  return byName >= 0 ? byName : 0;
}

function buildColumns(text: string, isJsonl: boolean): SchemaCol[] {
  const sampled = sampleColumns(text, isJsonl).filter((c) => c.name);
  const pkIndex = detectPrimaryKey(sampled);
  return sampled.map((c, i) => ({ name: c.name, type: inferType(c.values), is_primary_key: i === pkIndex }));
}

function validateSchema(columns: SchemaCol[]): string[] {
  const errors: string[] = [];
  const pkCount = columns.filter((c) => c.is_primary_key).length;
  if (pkCount === 0) errors.push("Select a primary-key column.");
  if (pkCount > 1) errors.push("Only one column can be the primary key.");
  if (!columns.some((c) => c.type === "text" && !c.is_primary_key)) errors.push("Mark at least one non-key column as text (the content to analyze).");
  const dupes = [...new Set(columns.map((c) => c.name).filter((n, i, a) => a.indexOf(n) !== i))];
  if (dupes.length) errors.push(`Duplicate column names: ${dupes.join(", ")}.`);
  return errors;
}
