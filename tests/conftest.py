import numpy as np
import pytest

from reviewscope_ml.pipelines.artifacts import ClusterInfo, RunArtifacts


def make_cluster(cid: int, size: int, label: str = None, micro=()) -> ClusterInfo:
    return ClusterInfo(
        cluster_id=cid,
        size=size,
        label=label or f"label {cid}",
        summary=f"summary {cid}",
        label_source="terms_fallback",
        top_terms=[["alpha", 1.0], ["beta", 0.5]],
        tfidf_terms=[["alpha", 2.0]],
        word_frequencies={"alpha": 3},
        sample_doc_ids=[f"d{cid}_0"],
        mean_stars=3.0,
        micro_cluster_ids=list(micro),
    )


@pytest.fixture
def small_run() -> RunArtifacts:
    """12 docs: cluster 0 (5 docs), cluster 1 (4), noise (3)."""
    labels = np.array([0] * 5 + [1] * 4 + [-1] * 3)
    n = len(labels)
    return RunArtifacts(
        run_name="testrun",
        manifest={"run_name": "testrun", "variant": "custom_hdbscan", "sample_size": n},
        doc_ids=[f"doc{i}" for i in range(n)],
        labels=labels,
        coords_2d=np.arange(n * 2, dtype=float).reshape(n, 2),
        coords_3d=np.arange(n * 3, dtype=float).reshape(n, 3),
        clusters={0: make_cluster(0, 5), 1: make_cluster(1, 4)},
        metrics={"n_clusters": 2, "noise_ratio": 0.25},
    )


@pytest.fixture
def two_stage_run() -> RunArtifacts:
    """Macro cluster 0 made of micro 10+11, macro 1 made of micro 12."""
    labels = np.array([0, 0, 0, 0, 1, 1, -1])
    micro = np.array([10, 10, 11, 11, 12, 12, -1])
    n = len(labels)
    return RunArtifacts(
        run_name="twostage",
        manifest={"run_name": "twostage", "variant": "two_stage", "sample_size": n},
        doc_ids=[f"doc{i}" for i in range(n)],
        labels=labels,
        coords_2d=np.zeros((n, 2)),
        coords_3d=np.zeros((n, 3)),
        clusters={
            0: make_cluster(0, 4, micro=(10, 11)),
            1: make_cluster(1, 2, micro=(12,)),
        },
        metrics={},
        micro_labels=micro,
    )
