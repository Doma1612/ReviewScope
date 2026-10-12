import { EmbeddingPoint } from "./api";


export function sentimentLabel(score: number | null | undefined): string | null {
  if (score == null) return null;
  const tone = score > 0.05 ? "positive" : score < -0.05 ? "negative" : "neutral";
  return `${tone} (${score.toFixed(2)})`;
}

export function sentimentSummary(avg: number | null | undefined, count: number, total: number): string {
  if (avg == null || count === 0) return "sentiment n/a";
  return `sentiment ${avg.toFixed(2)} · ${count} of ${total} scored`;
}

function escapeHtml(value: string): string {
  return value.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
}

export function hoverHtml(point: EmbeddingPoint): string {
  const label = point.cluster_label ?? "noise";
  const pk = point.primary_key_value ?? point.document_id;
  const senti = sentimentLabel(point.sentiment_score);
  const sentiPart = senti ? ` · ${escapeHtml(senti)}` : "";
  return `<b>${escapeHtml(label)}</b><br>${escapeHtml(point.snippet ?? "")}<br><i>${escapeHtml(pk)}</i>${sentiPart}`;
}

export function hoverTitle(point: EmbeddingPoint): string {
  const label = point.cluster_label ?? "noise";
  const pk = point.primary_key_value ?? point.document_id;
  const senti = sentimentLabel(point.sentiment_score);
  const lines = [label];
  if (point.snippet) lines.push(point.snippet);
  lines.push(senti ? `${pk} · ${senti}` : pk);
  return lines.join("\n");
}
