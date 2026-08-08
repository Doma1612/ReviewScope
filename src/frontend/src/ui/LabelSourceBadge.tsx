// Label provenance badge. Distinguishes how a cluster's label was produced:
//   ollama:<model>  → AI-generated (LLM)
//   terms_fallback  → keyword fallback (LLM was unavailable)
//   hitl_override   → human-edited
//   hitl_approved   → human-approved (machine label confirmed)
// So a keyword guess is never mistaken for an AI label or a vetted human one.
type Provenance = { text: string; tier: string; title: string };

function describe(labelSource: string): Provenance {
  if (labelSource.startsWith("ollama:")) {
    return { text: "AI", tier: "ai", title: `AI-generated label (${labelSource})` };
  }
  switch (labelSource) {
    case "hitl_override":
      return { text: "Edited", tier: "human", title: "Label edited by a person" };
    case "hitl_approved":
      return { text: "Approved", tier: "human", title: "Machine label approved by a person" };
    case "terms_fallback":
    default:
      return { text: "Keywords", tier: "fallback", title: "Keyword fallback — no AI label was generated" };
  }
}

export function LabelSourceBadge({ labelSource }: { labelSource: string }) {
  if (!labelSource) return null;
  const { text, tier, title } = describe(labelSource);
  return (
    <span className={`label-source-badge ${tier}`} title={title}>{text}</span>
  );
}
