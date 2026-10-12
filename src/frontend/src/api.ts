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
  unit: "document" | "sentence";
  retrieval_ready: boolean;
};

export type PipelineStatus = {
  project_id: string;
  status: Project["status"];
  jobs: { step: string; status: string; message: string | null }[];
};

export type EmbeddingPoint = {
  document_id: string;
  // Sentence points identify a segment; document_id identifies its parent review.
  segment_id: string | null;
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
  size: number; // distinct parent reviews
  n_mentions: number; // segment mentions (equal to size for document runs)
  sentiment_avg: number | null;
  sentiment_count: number;
  mean_stars: number | null;
  cohesion: number | null;
  sample_docs: { id: string; text: string }[];
};
export type ChatMessage = { role: "user" | "assistant"; content: string };
export type ChatSource = {
  document_id: string;
  primary_key_value: string;
  matched_field: string | null;
  matched_value: string | null;
  rating: number | null;
  snippet: string;
  full_text: string | null;
  raw_data: Record<string, unknown>;
};
export type ChatResult = {
  document_id: string;
  primary_key_value: string;
  rating: number | null;
  snippet: string;
};
export type ChatToolState = Record<string, unknown>;
export type ProjectChatSource = {
  citation_id: string;
  unit: "document" | "segment" | "cluster";
  document_id: string | null;
  segment_id: string | null;
  primary_key_value: string | null;
  cluster_id: string;
  cluster_label: string;
  rating: number | null;
  snippet: string;
  parent_text: string | null;
  distance: number | null;
  size: number | null;
  n_mentions: number | null;
  sentiment_avg: number | null;
  mean_stars: number | null;
};
export type ProjectChatStreamHandlers = {
  onSources?: (sources: ProjectChatSource[]) => void;
  onToken?: (content: string) => void;
  onDone?: (model: string) => void;
  onError?: (message: string) => void;
};
export type ChatStreamHandlers = {
  onSources?: (sources: ChatSource[]) => void;
  onResults?: (results: ChatResult[]) => void;
  onToolState?: (toolState: ChatToolState | null) => void;
  onToken?: (content: string) => void;
  onDone?: (model: string) => void;
  onError?: (message: string) => void;
};
export type ClusterMembership = { cluster_id: string; cluster_label: string; mention_count: number; share: number };
export type DocumentItem = {
  id: string;
  primary_key_value: string;
  text: string;
  raw_data: Record<string, unknown>;
  cluster_id: string | null;
  sentiment_score: number | null;
  memberships?: ClusterMembership[];
  primary_cluster_id?: string | null;
};
export type Member = { user_id: string; email: string; role: "owner" | "viewer" };
export type Models = { embedding_model: string; label_model: string; variant: string; simulated: boolean };
export type ProjectMetrics = { metrics: Record<string, unknown> | null; computed_at: string | null; stale: boolean };
export type SchemaColumn = { name: string; type: string; is_primary_key: boolean };
export type DocumentFilter = { column: string; op: "eq" | "gte" | "lte"; value: string; type: string };
export type ClusterEdit = {
  id: string;
  project_id: string;
  actor_id: string;
  action: string;
  created_at: string;
  cluster_id: string | null;
  target_cluster_id: string | null;
  document_id: string | null;
  segment_id: string | null;
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

async function requestNdjson(path: string, options: RequestInit, handlers: ChatStreamHandlers): Promise<void> {
  const response = await fetch(`${API_BASE}${path}`, {
    credentials: "include",
    headers: { "Content-Type": "application/json", ...options.headers },
    ...options,
  });
  if (response.status === 404 && path.endsWith("/stream")) {
    const fallback = await request<{ message: ChatMessage; model: string; sources: ChatSource[]; results?: ChatResult[]; tool_state?: ChatToolState | null }>(path.replace(/\/stream$/, ""), options);
    handlers.onSources?.(fallback.sources ?? []);
    handlers.onResults?.(fallback.results ?? []);
    handlers.onToolState?.(fallback.tool_state ?? null);
    handlers.onToken?.(fallback.message.content);
    handlers.onDone?.(fallback.model);
    return;
  }
  if (!response.ok) {
    const body = await response.json().catch(() => ({ detail: response.statusText }));
    throw new Error(body.detail ?? "Request failed");
  }
  if (!response.body) throw new Error("Streaming response is not available");

  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  while (true) {
    const { value, done } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    const lines = buffer.split("\n");
    buffer = lines.pop() ?? "";
    for (const line of lines) processChatEvent(line, handlers);
  }
  buffer += decoder.decode();
  if (buffer.trim()) processChatEvent(buffer, handlers);
}

function processChatEvent(line: string, handlers: ChatStreamHandlers) {
  if (!line.trim()) return;
  const event = JSON.parse(line) as { type: string; sources?: ChatSource[]; results?: ChatResult[]; tool_state?: ChatToolState | null; content?: string; model?: string; message?: string };
  if (event.type === "sources") handlers.onSources?.(event.sources ?? []);
  else if (event.type === "results") handlers.onResults?.(event.results ?? []);
  else if (event.type === "tool_state") handlers.onToolState?.(event.tool_state ?? null);
  else if (event.type === "token") handlers.onToken?.(event.content ?? "");
  else if (event.type === "done") handlers.onDone?.(event.model ?? "");
  else if (event.type === "error") handlers.onError?.(event.message ?? "Chat request failed");
}

async function requestProjectChatNdjson(path: string, options: RequestInit, handlers: ProjectChatStreamHandlers): Promise<void> {
  const response = await fetch(`${API_BASE}${path}`, {
    credentials: "include",
    headers: { "Content-Type": "application/json", ...options.headers },
    ...options,
  });
  if (response.status === 404 && path.endsWith("/stream")) {
    const fallbackResponse = await fetch(`${API_BASE}${path.replace(/\/stream$/, "")}`, {
      credentials: "include",
      headers: { "Content-Type": "application/json", ...options.headers },
      ...options,
    });
    if (!fallbackResponse.ok) {
      const body = await fallbackResponse.json().catch(() => ({ detail: fallbackResponse.statusText }));
      throw new Error(projectChatErrorMessage(body, fallbackResponse.statusText));
    }
    const fallback = await fallbackResponse.json() as { message?: { content?: unknown }; model: string; sources: ProjectChatSource[] };
    handlers.onSources?.(fallback.sources ?? []);
    const content = projectTokenText(fallback.message?.content);
    if (content == null) handlers.onError?.("The chat server returned an invalid response message.");
    else {
      handlers.onToken?.(content);
      handlers.onDone?.(fallback.model);
    }
    return;
  }
  if (!response.ok) {
    const body = await response.json().catch(() => ({ detail: response.statusText }));
    throw new Error(projectChatErrorMessage(body, response.statusText));
  }
  if (!response.body) throw new Error("Streaming response is not available");
  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  while (true) {
    const { value, done } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    const lines = buffer.split("\n");
    buffer = lines.pop() ?? "";
    for (const line of lines) processProjectChatEvent(line, handlers);
  }
  buffer += decoder.decode();
  if (buffer.trim()) processProjectChatEvent(buffer, handlers);
}

function projectChatErrorMessage(body: unknown, fallback: string): string {
  if (!body || typeof body !== "object" || !("detail" in body)) return fallback || "Project chat request failed";
  const detail = (body as { detail?: unknown }).detail;
  if (typeof detail === "string") return detail;
  if (Array.isArray(detail)) {
    const messages = detail.map((item) => {
      if (!item || typeof item !== "object") return null;
      const issue = item as { loc?: unknown; msg?: unknown };
      return typeof issue.msg === "string" ? `${Array.isArray(issue.loc) ? issue.loc.join(".") + ": " : ""}${issue.msg}` : null;
    }).filter((message): message is string => Boolean(message));
    if (messages.length) return messages.join("; ");
  }
  return fallback || "Project chat request failed";
}

function projectTokenText(value: unknown): string | null {
  if (typeof value === "string") return value;
  if (!value || typeof value !== "object" || !("content" in value)) return null;
  const nested = (value as { content?: unknown }).content;
  return typeof nested === "string" ? nested : null;
}

function processProjectChatEvent(line: string, handlers: ProjectChatStreamHandlers) {
  if (!line.trim()) return;
  const event = JSON.parse(line) as { type: string; sources?: ProjectChatSource[]; content?: unknown; model?: string; message?: string };
  if (event.type === "sources") handlers.onSources?.(event.sources ?? []);
  else if (event.type === "token") {
    const content = projectTokenText(event.content);
    if (content == null) handlers.onError?.("The chat server returned an invalid response token.");
    else handlers.onToken?.(content);
  }
  else if (event.type === "done") handlers.onDone?.(event.model ?? "");
  else if (event.type === "error") handlers.onError?.(event.message ?? "Chat request failed");
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
  embeddings: (projectId: string, sample?: number) =>
    request<EmbeddingPoint[]>(`/api/projects/${projectId}/embeddings${sample != null ? `?sample=${sample}` : ""}`),
  embeddingStats: (projectId: string) => request<{ total: number; noise: number }>(`/api/projects/${projectId}/embeddings/stats`),
  clusters: (projectId: string) => request<Cluster[]>(`/api/projects/${projectId}/clusters`),
  cluster: (projectId: string, clusterId: string) => request<Cluster>(`/api/projects/${projectId}/clusters/${clusterId}`),
  documents: (projectId: string, params: { clusterId?: string; limit?: number; offset?: number; filters?: DocumentFilter[] } = {}) => {
    const query = new URLSearchParams();
    if (params.clusterId) query.set("cluster_id", params.clusterId);
    if (params.limit != null) query.set("limit", String(params.limit));
    if (params.offset != null) query.set("offset", String(params.offset));
    if (params.filters?.length) query.set("filters", JSON.stringify(params.filters));
    const qs = query.toString();
    return request<DocumentItem[]>(`/api/projects/${projectId}/documents${qs ? `?${qs}` : ""}`);
  },
  documentsCount: (projectId: string, params: { clusterId?: string; filters?: DocumentFilter[] } = {}) => {
    const query = new URLSearchParams();
    if (params.clusterId) query.set("cluster_id", params.clusterId);
    if (params.filters?.length) query.set("filters", JSON.stringify(params.filters));
    const qs = query.toString();
    return request<{ total: number }>(`/api/projects/${projectId}/documents/count${qs ? `?${qs}` : ""}`);
  },
  clusterDocuments: (projectId: string, clusterId: string) => request<DocumentItem[]>(`/api/projects/${projectId}/clusters/${clusterId}/documents`),
  clusterChat: (projectId: string, clusterId: string, message: string, history: ChatMessage[], activeSourceIds: string[], lastToolState?: ChatToolState | null) =>
    request<{ message: ChatMessage; model: string; sources: ChatSource[]; results: ChatResult[]; tool_state: ChatToolState | null }>(`/api/projects/${projectId}/clusters/${clusterId}/chat`, { method: "POST", body: JSON.stringify({ message, history, active_source_ids: activeSourceIds, last_tool_state: lastToolState ?? null }) }),
  clusterChatStream: (projectId: string, clusterId: string, message: string, history: ChatMessage[], activeSourceIds: string[], lastToolState: ChatToolState | null, handlers: ChatStreamHandlers) =>
    requestNdjson(`/api/projects/${projectId}/clusters/${clusterId}/chat/stream`, { method: "POST", body: JSON.stringify({ message, history, active_source_ids: activeSourceIds, last_tool_state: lastToolState }) }, handlers),
  projectChatStream: (projectId: string, message: string, history: ChatMessage[], handlers: ProjectChatStreamHandlers) =>
    requestProjectChatNdjson(`/api/projects/${projectId}/chat/stream`, { method: "POST", body: JSON.stringify({ message, history }) }, handlers),
  reassignReview: (projectId: string, documentId: string, clusterId: string | null) =>
    request<DocumentItem>(`/api/projects/${projectId}/documents/${documentId}`, { method: "PATCH", body: JSON.stringify({ cluster_id: clusterId }) }),
  bulkReassign: (projectId: string, documentIds: string[], clusterId: string | null) =>
    request<{ moved: number }>(`/api/projects/${projectId}/documents/reassign`, { method: "POST", body: JSON.stringify({ document_ids: documentIds, cluster_id: clusterId }) }),
  reassignSegment: (projectId: string, segmentId: string, clusterId: string | null) =>
    request<EmbeddingPoint>(`/api/projects/${projectId}/segments/${segmentId}`, { method: "PATCH", body: JSON.stringify({ cluster_id: clusterId }) }),
  bulkReassignSegments: (projectId: string, segmentIds: string[], clusterId: string | null) =>
    request<{ moved: number }>(`/api/projects/${projectId}/segments/reassign`, { method: "POST", body: JSON.stringify({ segment_ids: segmentIds, cluster_id: clusterId }) }),
  mergeClusters: (projectId: string, sourceIds: string[], targetId: string) =>
    request<Cluster>(`/api/projects/${projectId}/clusters/merge`, { method: "POST", body: JSON.stringify({ source_ids: sourceIds, target_id: targetId }) }),
  createClusterFromSegments: (projectId: string, segmentIds: string[], label: string) =>
    request<Cluster>(`/api/projects/${projectId}/clusters/from-selection`, { method: "POST", body: JSON.stringify({ segment_ids: segmentIds, label }) }),
  updateCluster: (projectId: string, clusterId: string, changes: { label?: string; approve?: boolean; markJunk?: boolean }) =>
    request<Cluster | void>(`/api/projects/${projectId}/clusters/${clusterId}`, { method: "PATCH", body: JSON.stringify({ label: changes.label, approve: changes.approve, mark_junk: changes.markJunk }) }),
  deleteCluster: (projectId: string, clusterId: string) =>
    request<void>(`/api/projects/${projectId}/clusters/${clusterId}`, { method: "DELETE" }),
  getSchema: (projectId: string) => request<{ columns: SchemaColumn[] }>(`/api/projects/${projectId}/schema`),
  edits: (projectId: string) => request<ClusterEdit[]>(`/api/projects/${projectId}/edits`),
  projectMetrics: (projectId: string) => request<ProjectMetrics>(`/api/projects/${projectId}/metrics`),
  members: (projectId: string) => request<Member[]>(`/api/projects/${projectId}/members`),
  addMember: (projectId: string, email: string) => request<Member>(`/api/projects/${projectId}/members`, { method: "POST", body: JSON.stringify({ email, role: "viewer" }) }),
  updateMember: (projectId: string, userId: string, role: Member["role"]) => request<Member>(`/api/projects/${projectId}/members/${userId}`, { method: "PATCH", body: JSON.stringify({ role }) }),
  removeMember: (projectId: string, userId: string) => request<void>(`/api/projects/${projectId}/members/${userId}`, { method: "DELETE" }),
  models: () => request<Models>("/api/models"),
};
