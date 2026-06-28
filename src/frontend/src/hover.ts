import { EmbeddingPoint } from "./api";

// Shared scatter-hover builders. Both the Plotly scatter (ProjectView) and
// the CSS point-cloud (DeckDashboard) show cluster label + snippet + primary key
// + sentiment instead of a bare document UUID. Mirrors the prototype hover string
// in src/reviewscope_ml/hitl/app.py.

export function sentimentLabel(score: number | null | undefined): string | null {
  if (score == null) return null;
  const tone = score > 0.05 ? "positive" : score < -0.05 ? "negative" : "neutral";
  return `${tone} (${score.toFixed(2)})`;
}

// Honest one-line sentiment summary for a cluster: the rounded mean plus how many
// of its documents were actually scored, so a mean over a handful of docs isn't
// read as covering the whole cluster. "sentiment n/a" when nothing was scored.
export function sentimentSummary(avg: number | null | undefined, count: number, total: number): string {
  if (avg == null || count === 0) return "sentiment n/a";
  return `sentiment ${avg.toFixed(2)} · ${count} of ${total} scored`;
}

function escapeHtml(value: string): string {
  return value.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
}

// Plotly `text` entry: rendered with `hovertemplate: "%{text}<extra></extra>"`.
export function hoverHtml(point: EmbeddingPoint): string {
  const label = point.cluster_label ?? "noise";
  const pk = point.primary_key_value ?? point.document_id;
  const senti = sentimentLabel(point.sentiment_score);
  const sentiPart = senti ? ` · ${escapeHtml(senti)}` : "";
  return `<b>${escapeHtml(label)}</b><br>${escapeHtml(point.snippet ?? "")}<br><i>${escapeHtml(pk)}</i>${sentiPart}`;
}

// Plain-text variant for the `title` attribute of a DOM point.
export function hoverTitle(point: EmbeddingPoint): string {
  const label = point.cluster_label ?? "noise";
  const pk = point.primary_key_value ?? point.document_id;
  const senti = sentimentLabel(point.sentiment_score);
  const lines = [label];
  if (point.snippet) lines.push(point.snippet);
  lines.push(senti ? `${pk} · ${senti}` : pk);
  return lines.join("\n");
}
