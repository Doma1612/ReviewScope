import { FormEvent, KeyboardEvent, ReactNode, useEffect, useRef, useState } from "react";
import { Link } from "react-router-dom";

import { api, ChatMessage, Cluster, ProjectChatSource } from "../api";

type DisplayMessage = ChatMessage & { sources?: ProjectChatSource[]; sourcesVisible?: boolean };

function escapeRegex(value: string) {
  return value.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
}

function renderClusterReferences(text: string, projectId: string, clusters: Cluster[], keyPrefix: string): ReactNode[] {
  const references = clusters.flatMap((cluster) => [cluster.id, cluster.label]).filter(Boolean).sort((a, b) => b.length - a.length);
  if (!references.length) return [text];
  const clustersByReference = new Map<string, Cluster>();
  for (const cluster of clusters) {
    clustersByReference.set(cluster.id.toLowerCase(), cluster);
    clustersByReference.set(cluster.label.toLowerCase(), cluster);
  }
  return text.split(new RegExp(`(${references.map(escapeRegex).join("|")})`, "gi")).filter(Boolean).map((part, index) => {
    const cluster = clustersByReference.get(part.toLowerCase());
    return cluster
      ? <Link key={`${keyPrefix}-${index}`} to={`/projects/${projectId}/clusters/${cluster.id}`}>{part}</Link>
      : part;
  });
}

function renderInlineMarkdown(text: string, projectId: string, clusters: Cluster[]): ReactNode[] {
  return text.split(/(\*\*[^*]+\*\*|`[^`]+`)/g).filter(Boolean).map((part, index) => {
    if (part.startsWith("**") && part.endsWith("**")) return <strong key={index}>{renderClusterReferences(part.slice(2, -2), projectId, clusters, `bold-${index}`)}</strong>;
    if (part.startsWith("`") && part.endsWith("`")) return <code key={index}>{part.slice(1, -1)}</code>;
    return <span key={index}>{renderClusterReferences(part, projectId, clusters, `text-${index}`)}</span>;
  });
}

function ChatText({ content, pending, projectId, clusters }: { content: string; pending?: boolean; projectId: string; clusters: Cluster[] }) {
  const lines = content.replace(/```(?:markdown)?/gi, "").replace(/```/g, "").split("\n").filter((line) => line.trim());
  const blocks: ReactNode[] = [];
  let bullets: string[] = [];
  const flushBullets = () => {
    if (!bullets.length) return;
    blocks.push(<ul key={`bullets-${blocks.length}`}>{bullets.map((line, index) => <li key={index}>{renderInlineMarkdown(line, projectId, clusters)}</li>)}</ul>);
    bullets = [];
  };
  for (const line of lines) {
    const bullet = line.match(/^\s*[-*]\s+(.+)$/);
    if (bullet) bullets.push(bullet[1]);
    else { flushBullets(); blocks.push(<p key={`p-${blocks.length}`}>{renderInlineMarkdown(line, projectId, clusters)}</p>); }
  }
  flushBullets();
  return <div className="chat-text">{blocks}{pending && <span className="typing-cursor" aria-hidden />}</div>;
}

export function ProjectChat({ projectId, available, clusters }: { projectId: string; available: boolean; clusters: Cluster[] }) {
  const [messages, setMessages] = useState<DisplayMessage[]>([]);
  const [draft, setDraft] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [pending, setPending] = useState(false);
  const [open, setOpen] = useState(false);
  const endRef = useRef<HTMLDivElement | null>(null);
  const projectRef = useRef(projectId);
  projectRef.current = projectId;

  useEffect(() => {
    setMessages([]); setDraft(""); setError(null); setPending(false); setOpen(false);
  }, [projectId]);
  useEffect(() => { endRef.current?.scrollIntoView({ behavior: "smooth", block: "end" }); }, [messages, pending, open]);

  const submit = async () => {
    const question = draft.trim();
    if (!question || pending || !available) return;
    // Do not let a stale malformed response poison the next request's Pydantic
    // history validation. Valid messages retain the full local conversation.
    const history = messages.slice(-8).flatMap(({ role, content }) => typeof content === "string" ? [{ role, content }] : []);
    const next = [...messages, { role: "user" as const, content: question }, { role: "assistant" as const, content: "", sources: [], sourcesVisible: false }];
    const assistantIndex = next.length - 1;
    const requestProjectId = projectId;
    setMessages(next); setDraft(""); setError(null); setPending(true);
    try {
      await api.projectChatStream(projectId, question, history, {
        onSources: (sources) => {
          if (projectRef.current !== requestProjectId) return;
          setMessages((current) => current.map((item, index) => index === assistantIndex ? { ...item, sources } : item));
        },
        onToken: (content) => {
          if (projectRef.current !== requestProjectId) return;
          if (typeof content !== "string") {
            setError("The chat server returned an invalid response token.");
            return;
          }
          setMessages((current) => current.map((item, index) => index === assistantIndex ? { ...item, content: item.content + content } : item));
        },
        onDone: () => {
          if (projectRef.current !== requestProjectId) return;
          setMessages((current) => current.map((item, index) => index === assistantIndex ? { ...item, sourcesVisible: true } : item));
        },
        onError: (message) => { if (projectRef.current === requestProjectId) setError(message); },
      });
    } catch (requestError) {
      if (projectRef.current !== requestProjectId) return;
      setMessages(messages); setDraft(question);
      setError(requestError instanceof Error ? requestError.message : "Chat request failed");
    } finally {
      if (projectRef.current === requestProjectId) setPending(false);
    }
  };
  const onSubmit = (event: FormEvent) => { event.preventDefault(); void submit(); };
  const onKeyDown = (event: KeyboardEvent<HTMLTextAreaElement>) => {
    if (event.key === "Enter" && !event.shiftKey) { event.preventDefault(); void submit(); }
  };

  return <aside className={`floating-chat ${open ? "open" : ""}`} aria-label="Project chat">
    {open ? <section className="floating-chat-panel">
      <div className="floating-chat-header">
        <div><h2>Ask this project</h2><span>Semantic search across all clusters</span></div>
        <button type="button" className="floating-chat-close" aria-label="Close project chat" onClick={() => setOpen(false)}>x</button>
      </div>
      <div className="chat-log" aria-live="polite">
        {messages.length === 0 && <p className="chat-empty">Ask about themes, patterns, risks, or contradictions across this project&apos;s clusters.</p>}
        {messages.map((message, index) => <div className={`chat-message ${message.role}`} key={`${message.role}-${index}`}>
          <strong>{message.role === "user" ? "You" : "ReviewScope"}</strong>
          <ChatText content={message.content} pending={pending && message.role === "assistant" && index === messages.length - 1} projectId={projectId} clusters={clusters} />
          {message.sourcesVisible && message.sources?.length ? <details className="chat-sources">
            <summary><span className="chat-sources-label">Evidence used</span> · {message.sources.length} source{message.sources.length === 1 ? "" : "s"}</summary>
            <div className="chat-sources-list">{message.sources.map((source) => <details key={source.citation_id}>
              <summary>[{source.citation_id}] {source.cluster_label}{source.unit === "cluster" ? ` · ${source.size ?? 0} reviews` : ` · Review ${source.primary_key_value ?? "n/a"}`}{source.rating != null ? ` · ${source.rating} stars` : ""}</summary>
              <p><Link to={`/projects/${projectId}/clusters/${source.cluster_id}`}>{source.cluster_label}</Link> · {source.snippet}</p>
              {source.unit === "segment" && source.parent_text && <details><summary>Full review context</summary><p>{source.parent_text}</p></details>}
            </details>)}</div>
          </details> : null}
        </div>)}
        <div ref={endRef} />
      </div>
      {error && <p className="chat-error">{error}</p>}
      {available ? <form className="chat-form" onSubmit={onSubmit}>
        <textarea value={draft} onChange={(event) => setDraft(event.target.value)} onKeyDown={onKeyDown} placeholder="Ask the project..." rows={2} />
        <button className="button primary" type="submit" disabled={!draft.trim() || pending}>Send</button>
      </form> : <p className="chat-error">Reprocess this project to enable semantic search.</p>}
    </section> : <button type="button" className="floating-chat-bubble" onClick={() => setOpen(true)}><span>Ask the project</span></button>}
  </aside>;
}
