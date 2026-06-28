export type Project = {
  id: string;
  name: string;
  owner_id: string;
  owner_email: string | null;
  status: "uploading" | "processing" | "ready" | "failed";
  doc_count: number;
  created_at: string;
  role: "owner" | "viewer";
  last_error: string | null;
};

export type PipelineStatus = {
  project_id: string;
  status: Project["status"];
  jobs: { step: string; status: string; message: string | null }[];
};

export type EmbeddingPoint = {
  document_id: string;
  cluster_id: string | null;
  x: number;
  y: number;
  z: number | null;
  snippet: string | null;
  primary_key_value: string | null;
  sentiment_score: number | null;
  cluster_label: string | null;
};
export type Cluster = {
  id: string;
  label: string;
  summary: string;
  label_source: string;
  top_terms: { term: string; score: number }[];
  word_frequencies: Record<string, number>;
  size: number;
  sentiment_avg: number | null;
  mean_stars: number | null;
  sample_docs: { id: string; text: string }[];
};
export type DocumentItem = { id: string; primary_key_value: string; text: string; raw_data: Record<string, unknown>; cluster_id: string | null; sentiment_score: number | null };
export type Member = { user_id: string; email: string; role: "owner" | "viewer" };
export type Models = { embedding_model: string; label_model: string; variant: string; simulated: boolean };
export type Health = { status: string };
export type SchemaColumn = { name: string; type: string; is_primary_key: boolean };
export type ClusterEdit = {
  id: string;
  project_id: string;
  actor_id: string;
  action: string;
  created_at: string;
  cluster_id: string | null;
  target_cluster_id: string | null;
  document_id: string | null;
  new_label: string | null;
  note: string | null;
  payload: Record<string, unknown>;
};

const API_BASE = import.meta.env.VITE_API_BASE_URL ?? "http://localhost:8000";

async function request<T>(path: string, options: RequestInit = {}): Promise<T> {
  const response = await fetch(`${API_BASE}${path}`, {
    credentials: "include",
    headers: options.body instanceof FormData ? options.headers : { "Content-Type": "application/json", ...options.headers },
    ...options,
  });
  if (!response.ok) {
    const body = await response.json().catch(() => ({ detail: response.statusText }));
    throw new Error(body.detail ?? "Request failed");
  }
  if (response.status === 204) return undefined as T;
  return response.json() as Promise<T>;
}

export const api = {
  register: (email: string, password: string) => request("/api/auth/register", { method: "POST", body: JSON.stringify({ email, password }) }),
  login: (email: string, password: string) => request("/api/auth/login", { method: "POST", body: JSON.stringify({ email, password }) }),
  logout: () => request("/api/auth/logout", { method: "POST" }),
  me: () => request<{ id: string; email: string }>("/api/auth/me"),
  projects: () => request<Project[]>("/api/projects"),
  project: (projectId: string) => request<Project>(`/api/projects/${projectId}`),
  updateProject: (projectId: string, name: string) => request<Project>(`/api/projects/${projectId}`, { method: "PATCH", body: JSON.stringify({ name }) }),
  deleteProject: (projectId: string) => request<void>(`/api/projects/${projectId}`, { method: "DELETE" }),
  uploadProject: (form: FormData) => request<Project>("/api/projects", { method: "POST", body: form }),
  pipelineStatus: (projectId: string) => request<PipelineStatus>(`/api/projects/${projectId}/pipeline/status`),
  embeddings: (projectId: string) => request<EmbeddingPoint[]>(`/api/projects/${projectId}/embeddings`),
  clusters: (projectId: string) => request<Cluster[]>(`/api/projects/${projectId}/clusters`),
  cluster: (projectId: string, clusterId: string) => request<Cluster>(`/api/projects/${projectId}/clusters/${clusterId}`),
  documents: (projectId: string, params: { clusterId?: string; limit?: number; offset?: number } = {}) => {
    const query = new URLSearchParams();
    if (params.clusterId) query.set("cluster_id", params.clusterId);
    if (params.limit != null) query.set("limit", String(params.limit));
    if (params.offset != null) query.set("offset", String(params.offset));
    const qs = query.toString();
    return request<DocumentItem[]>(`/api/projects/${projectId}/documents${qs ? `?${qs}` : ""}`);
  },
  clusterDocuments: (projectId: string, clusterId: string) => request<DocumentItem[]>(`/api/projects/${projectId}/clusters/${clusterId}/documents`),
  reassignDocument: (projectId: string, documentId: string, clusterId: string | null) =>
    request<DocumentItem>(`/api/projects/${projectId}/documents/${documentId}`, { method: "PATCH", body: JSON.stringify({ cluster_id: clusterId }) }),
  bulkReassign: (projectId: string, documentIds: string[], clusterId: string | null) =>
    request<{ moved: number }>(`/api/projects/${projectId}/documents/reassign`, { method: "POST", body: JSON.stringify({ document_ids: documentIds, cluster_id: clusterId }) }),
  createCluster: (projectId: string, label: string) =>
    request<Cluster>(`/api/projects/${projectId}/clusters`, { method: "POST", body: JSON.stringify({ label }) }),
  mergeClusters: (projectId: string, sourceIds: string[], targetId: string) =>
    request<Cluster>(`/api/projects/${projectId}/clusters/merge`, { method: "POST", body: JSON.stringify({ source_ids: sourceIds, target_id: targetId }) }),
  createClusterFromSelection: (projectId: string, documentIds: string[], label: string) =>
    request<Cluster>(`/api/projects/${projectId}/clusters/from-selection`, { method: "POST", body: JSON.stringify({ document_ids: documentIds, label }) }),
  updateCluster: (projectId: string, clusterId: string, changes: { label?: string; approve?: boolean; markJunk?: boolean }) =>
    request<Cluster | void>(`/api/projects/${projectId}/clusters/${clusterId}`, { method: "PATCH", body: JSON.stringify({ label: changes.label, approve: changes.approve, mark_junk: changes.markJunk }) }),
  deleteCluster: (projectId: string, clusterId: string) =>
    request<void>(`/api/projects/${projectId}/clusters/${clusterId}`, { method: "DELETE" }),
  getSchema: (projectId: string) => request<{ columns: SchemaColumn[] }>(`/api/projects/${projectId}/schema`),
  saveSchema: (projectId: string, columns: SchemaColumn[]) =>
    request<{ columns: SchemaColumn[] }>(`/api/projects/${projectId}/schema`, { method: "POST", body: JSON.stringify({ columns }) }),
  edits: (projectId: string) => request<ClusterEdit[]>(`/api/projects/${projectId}/edits`),
  members: (projectId: string) => request<Member[]>(`/api/projects/${projectId}/members`),
  addMember: (projectId: string, email: string) => request<Member>(`/api/projects/${projectId}/members`, { method: "POST", body: JSON.stringify({ email, role: "viewer" }) }),
  updateMember: (projectId: string, userId: string, role: Member["role"]) => request<Member>(`/api/projects/${projectId}/members/${userId}`, { method: "PATCH", body: JSON.stringify({ role }) }),
  removeMember: (projectId: string, userId: string) => request<void>(`/api/projects/${projectId}/members/${userId}`, { method: "DELETE" }),
  models: () => request<Models>("/api/models"),
  health: () => request<Health>("/api/health"),
};
