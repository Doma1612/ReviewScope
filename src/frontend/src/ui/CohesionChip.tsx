export function CohesionChip({ value, compact = false }: { value: number | null; compact?: boolean }) {
  if (value == null) return null;
  const tier = value >= 0.5 ? "high" : value >= 0.25 ? "medium" : "low";
  const label = tier === "high" ? "High" : tier === "medium" ? "Medium" : "Low";
  return (
    <span
      className={`cohesion-chip ${tier} ${compact ? "compact" : ""}`}
      title="Cohesion: mean cosine similarity of this cluster's documents to its centre (0–1). Higher means a tighter, more trustworthy cluster."
    >
      Cohesion: {label} ({value.toFixed(2)})
    </span>
  );
}
