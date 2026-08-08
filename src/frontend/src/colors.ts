// Shared cluster color palette. One source of truth so the Plotly scatter
// (ProjectView), the DOM point cloud (DeckDashboard), and the cluster legend all
// agree on which color a cluster gets.
//
// The palette is the Okabe–Ito colorblind-safe qualitative set (minus black, which
// disappears on the dark canvas). Noise (`cluster_id = null`) gets a dedicated
// reserved grey so it can never be mistaken for the first real cluster.
export const PALETTE = [
  "#56B4E9", // sky blue
  "#E69F00", // orange
  "#009E73", // bluish green
  "#F0E442", // yellow
  "#CC79A7", // reddish purple
  "#0072B2", // blue
  "#D55E00", // vermillion
  "#94d2bd", // teal
];

// Reserved grey for noise / unclustered points. Distinct from every PALETTE entry.
export const NOISE_COLOR = "#64748b";

export function clusterColor(index: number): string {
  return PALETTE[index % PALETTE.length];
}

// Color for a point given its cluster id and a cluster→index lookup. Noise (null)
// and orphan ids with no known cluster fall back to the reserved grey, so noise is
// always visually separable from real clusters.
export function pointColor(clusterId: string | null, clusterIndex: Map<string, number>): string {
  if (clusterId == null) return NOISE_COLOR;
  const index = clusterIndex.get(clusterId);
  return index == null ? NOISE_COLOR : clusterColor(index);
}
