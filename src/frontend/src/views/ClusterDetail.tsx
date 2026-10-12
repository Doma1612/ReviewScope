import { useQuery } from "@tanstack/react-query";
import { FormEvent, KeyboardEvent, ReactNode, useEffect, useMemo, useRef, useState } from "react";
import Plot from "react-plotly.js";
import { Link, useParams } from "react-router-dom";

import { api, ChatMessage, ChatResult, ChatSource, ChatToolState, DocumentItem } from "../api";
import { sentimentSummary } from "../hover";
import { CohesionChip } from "../ui/CohesionChip";
import { DocumentsTable } from "../ui/DocumentsTable";
import { LabelSourceBadge } from "../ui/LabelSourceBadge";
import { StarRating } from "../ui/StarRating";
import { WordCloud } from "../ui/WordCloud";

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

type DisplayChatMessage = ChatMessage & { sources?: ChatSource[]; results?: ChatResult[] };

function renderInlineMarkdown(text: string): ReactNode[] {
  return text.split(/(\*\*[^*]+\*\*|`[^`]+`)/g).filter(Boolean).map((part, index) => {
    if (part.startsWith("**") && part.endsWith("**")) return <strong key={index}>{part.slice(2, -2)}</strong>;
    if (part.startsWith("`") && part.endsWith("`")) return <code key={index}>{part.slice(1, -1)}</code>;
    return part;
  });
}

function ChatText({ content, pending }: { content: string; pending?: boolean }) {
  const lines = content
    .replace(/```(?:markdown)?/gi, "")
    .replace(/```/g, "")
    .split("\n")
    .filter((line) => line.trim() !== "");
  const blocks: ReactNode[] = [];
  let bullets: string[] = [];

  const flushBullets = () => {
    if (!bullets.length) return;
    blocks.push(<ul key={`bullets-${blocks.length}`}>{bullets.map((line, index) => <li key={index}>{renderInlineMarkdown(line)}</li>)}</ul>);
    bullets = [];
  };

  for (const line of lines) {
    const bullet = line.match(/^\s*[-*]\s+(.+)$/);
    if (bullet) {
      bullets.push(bullet[1]);
      continue;
    }
    flushBullets();
    blocks.push(<p key={`p-${blocks.length}`}>{renderInlineMarkdown(line)}</p>);
  }
  flushBullets();

  return <div className="chat-text">{blocks}{pending && <span className="typing-cursor" aria-hidden />}</div>;
}

function resultAnalysisText(content: string) {
  const lines = content.split("\n").slice(1);
  const kept = lines.filter((line) => {
    const trimmed = line.trim();
    if (!trimmed) return true;
    if (/^\d+\.\s+Review\s+/i.test(trimmed)) return false;
    if (/^(Rating|Text):/i.test(trimmed)) return false;
    return true;
  }).join("\n").trim();
  return kept;
}

export function ClusterDetail() {
  const { projectId = "", clusterId = "" } = useParams();
  const project = useQuery({ queryKey: ["project", projectId], queryFn: () => api.project(projectId) });
  const cluster = useQuery({ queryKey: ["cluster", projectId, clusterId], queryFn: () => api.cluster(projectId, clusterId) });
  const clusters = useQuery({ queryKey: ["clusters", projectId], queryFn: () => api.clusters(projectId) });
  const documents = useQuery({ queryKey: ["cluster-docs", projectId, clusterId], queryFn: () => api.clusterDocuments(projectId, clusterId) });

  const [editMode, setEditMode] = useState(false);
  const [chatMessages, setChatMessages] = useState<DisplayChatMessage[]>([]);
  const [chatDraft, setChatDraft] = useState("");
  const [chatError, setChatError] = useState<string | null>(null);
  const [chatPending, setChatPending] = useState(false);
  const [chatOpen, setChatOpen] = useState(false);
  const [activeSourceIds, setActiveSourceIds] = useState<string[]>([]);
  const [lastToolState, setLastToolState] = useState<ChatToolState | null>(null);
  const chatEndRef = useRef<HTMLDivElement | null>(null);
  const chatClusterIdRef = useRef(clusterId);
  chatClusterIdRef.current = clusterId;
  const isOwner = project.data?.role === "owner";
  const isSentence = project.data?.unit === "sentence";
  const canEdit = isOwner && isSentence;

  const buckets = useMemo(() => sentimentBuckets(documents.data ?? []), [documents.data]);
  const hasSentiment = buckets.negative + buckets.neutral + buckets.positive > 0;

  useEffect(() => {
    setChatMessages([]);
    setActiveSourceIds([]);
    setLastToolState(null);
    setChatPending(false);
    setChatError(null);
  }, [clusterId]);

  useEffect(() => {
    chatEndRef.current?.scrollIntoView({ behavior: "smooth", block: "end" });
  }, [chatMessages, chatPending, chatOpen]);

  const sendChat = async (event: FormEvent) => {
    event.preventDefault();
    void submitChat();
  };

  const submitChat = async () => {
    const question = chatDraft.trim();
    if (!question || chatPending) return;
    const history = chatMessages.map(({ role, content }) => ({ role, content }));
    const nextMessages: DisplayChatMessage[] = [...chatMessages, { role: "user", content: question }, { role: "assistant", content: "", sources: [] }];
    const assistantIndex = nextMessages.length - 1;
    setChatMessages(nextMessages);
    setChatDraft("");
    setChatError(null);
    setChatPending(true);
    const requestClusterId = clusterId;
    try {
      await api.clusterChatStream(projectId, clusterId, question, history, activeSourceIds, lastToolState, {
        onSources: (sources) => {
          if (chatClusterIdRef.current !== requestClusterId) return;
          setChatMessages((messages) => messages.map((message, index) => index === assistantIndex ? { ...message, sources } : message));
          if (sources.length) setActiveSourceIds(sources.map((source) => source.document_id));
        },
        onResults: (results) => {
          if (chatClusterIdRef.current !== requestClusterId) return;
          setChatMessages((messages) => messages.map((message, index) => index === assistantIndex ? { ...message, results } : message));
        },
        onToolState: (toolState) => {
          if (chatClusterIdRef.current !== requestClusterId) return;
          if (toolState) setLastToolState(toolState);
        },
        onToken: (content) => {
          if (chatClusterIdRef.current !== requestClusterId) return;
          setChatMessages((messages) => messages.map((message, index) => index === assistantIndex ? { ...message, content: message.content + content } : message));
        },
        onError: (message) => {
          if (chatClusterIdRef.current !== requestClusterId) return;
          setChatError(message);
        },
      });
    } catch (error) {
      if (chatClusterIdRef.current !== requestClusterId) return;
      setChatMessages(chatMessages);
      setChatDraft(question);
      setChatError(error instanceof Error ? error.message : "Chat request failed");
    } finally {
      if (chatClusterIdRef.current === requestClusterId) setChatPending(false);
    }
  };

  const sendChatOnEnter = (event: KeyboardEvent<HTMLTextAreaElement>) => {
    if (event.key !== "Enter" || event.shiftKey) return;
    event.preventDefault();
    void submitChat();
  };

  return (
    <main className="page">
      <Link to={`/projects/${projectId}`}>Back to cluster view</Link>
      <section className="page-header">
        <div><h1 className="cluster-detail-title">{cluster.data?.label ?? "Cluster"}{cluster.data && <LabelSourceBadge labelSource={cluster.data.label_source} />}</h1><p className="cluster-detail-meta">{cluster.data?.size ?? 0} {isSentence ? "reviews" : "documents"}{isSentence && cluster.data && cluster.data.n_mentions > cluster.data.size ? ` · ${cluster.data.n_mentions.toLocaleString()} mentions` : ""} · {cluster.data ? sentimentSummary(cluster.data.sentiment_avg, cluster.data.sentiment_count, cluster.data.size) : "sentiment n/a"}{cluster.data?.mean_stars != null && <> · <StarRating value={cluster.data.mean_stars} /></>}{cluster.data?.cohesion != null && <CohesionChip value={cluster.data.cohesion} />}</p></div>
        {canEdit && <button className={`button ${editMode ? "primary" : "secondary"}`} onClick={() => setEditMode((prev) => !prev)} type="button">{editMode ? "Done editing" : "Edit documents"}</button>}
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
            layout={{ autosize: true, height: 280, margin: { l: 40, r: 10, b: 40, t: 10 }, paper_bgcolor: "transparent", plot_bgcolor: "transparent", yaxis: { title: { text: "documents" } } }}
            useResizeHandler
            style={{ width: "100%" }}
          />
        </section>
      )}
      <section className="card"><h2>{isSentence ? "Reviews" : "Documents"}</h2><DocumentsTable projectId={projectId} clusterId={clusterId} editable={Boolean(editMode && canEdit)} clusters={clusters.data ?? []} sentence={isSentence} /></section>
      <aside className={`floating-chat ${chatOpen ? "open" : ""}`} aria-label="Cluster chat">
        {chatOpen ? (
          <section className="floating-chat-panel">
            <div className="floating-chat-header">
              <div><h2>Ask this cluster</h2><span>Ollama</span></div>
              <button type="button" className="floating-chat-close" aria-label="Close cluster chat" onClick={() => setChatOpen(false)}>x</button>
            </div>
            <div className="chat-log" aria-live="polite">
              {chatMessages.length === 0 && <p className="chat-empty">Ask about themes, evidence, risks, or contradictions in this cluster.</p>}
              {chatMessages.map((message, index) => (
                <div className={`chat-message ${message.role}`} key={`${message.role}-${index}`}>
                  <strong>{message.role === "user" ? "You" : "ReviewScope"}</strong>
                  {message.results?.length ? (
                    <div className="chat-results">
                      <p className="chat-results-title">{message.content.split("\n", 1)[0]}</p>
                      {resultAnalysisText(message.content) && <ChatText content={resultAnalysisText(message.content)} />}
                      {message.results.map((result, resultIndex) => (
                        <article className="chat-result-card" key={result.document_id}>
                          <div className="chat-result-head">
                            <strong>{resultIndex + 1}. Review {result.primary_key_value}</strong>
                            <span>{result.rating != null ? `${result.rating} stars` : "rating n/a"}</span>
                          </div>
                          <p>{result.snippet}</p>
                        </article>
                      ))}
                    </div>
                  ) : (
                    <ChatText content={message.content} pending={chatPending && message.role === "assistant" && index === chatMessages.length - 1} />
                  )}
                  {message.sources && message.sources.length > 0 && !message.results?.length && (
                    <div className="chat-sources">
                      <span className="chat-sources-label">Evidence used</span>
                      {message.sources.map((source) => (
                        <details key={source.document_id}>
                          <summary>Review {source.primary_key_value}{source.rating != null ? ` · ${source.rating} stars` : ""}</summary>
                          <p>{source.snippet}</p>
                        </details>
                      ))}
                    </div>
                  )}
                </div>
              ))}
              <div ref={chatEndRef} />
            </div>
            {chatError && <p className="chat-error">{chatError}</p>}
            <form className="chat-form" onSubmit={sendChat}>
              <textarea value={chatDraft} onChange={(event) => setChatDraft(event.target.value)} onKeyDown={sendChatOnEnter} placeholder="Ask the cluster..." rows={2} />
              <button className="button primary" type="submit" disabled={!chatDraft.trim() || chatPending}>Send</button>
            </form>
          </section>
        ) : (
          <button type="button" className="floating-chat-bubble" onClick={() => setChatOpen(true)}>
            <span>Ask the cluster</span>
          </button>
        )}
      </aside>
    </main>
  );
}
