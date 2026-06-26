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

export type EmbeddingPoint = { document_id: string; cluster_id: string | null; x: number; y: number; z: number | null };
export type Cluster = {
  id: string;
  label: string;
  summary: string;
  top_terms: { term: string; score: number }[];
  word_frequencies: Record<string, number>;
  size: number;
  sentiment_avg: number | null;
  sample_docs: { id: string; text: string }[];
};
export type DocumentItem = { id: string; primary_key_value: string; text: string; raw_data: Record<string, unknown>; cluster_id: string | null; sentiment_score: number | null };
export type Member = { user_id: string; email: string; role: "owner" | "viewer" };

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
  documents: (projectId: string, clusterId?: string) => request<DocumentItem[]>(`/api/projects/${projectId}/documents${clusterId ? `?cluster_id=${clusterId}` : ""}`),
  clusterDocuments: (projectId: string, clusterId: string) => request<DocumentItem[]>(`/api/projects/${projectId}/clusters/${clusterId}/documents`),
  members: (projectId: string) => request<Member[]>(`/api/projects/${projectId}/members`),
  addMember: (projectId: string, email: string) => request<Member>(`/api/projects/${projectId}/members`, { method: "POST", body: JSON.stringify({ email, role: "viewer" }) }),
  removeMember: (projectId: string, userId: string) => request<void>(`/api/projects/${projectId}/members/${userId}`, { method: "DELETE" }),
};
