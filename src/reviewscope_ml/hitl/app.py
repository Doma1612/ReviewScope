"""
HITL review GUI (Streamlit).

Launch from the repo root::

    streamlit run src/reviewscope_ml/hitl/app.py

Deliberately decoupled from the React frontend: the *feedback JSONL format*
(``feedback.py``) is the contract, this app is just the cheapest possible way
for a reviewer to produce it. It loads a finished run's artifacts, shows each
cluster with its label, terms, random samples and the 2-D scatter, and writes
every action append-only to ``data/feedback/``.

This file is the one place in the package allowed to import streamlit, and it
is an entry point — never imported by pipeline code.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import streamlit as st

# Entry-point bootstrapping: streamlit runs this file as a script, so the
# package import path must be set up explicitly when not pip-installed.
_SRC = Path(__file__).resolve().parents[2]
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from reviewscope_ml.core.config import load_config
from reviewscope_ml.data.ingest import load_benchmark
from reviewscope_ml.hitl.feedback import FeedbackRecord, append_record, session_file
from reviewscope_ml.pipelines.artifacts import load_run, run_is_complete

st.set_page_config(page_title="ReviewScope HITL review", layout="wide")


@st.cache_resource
def _load_run(run_dir: str):
    return load_run(Path(run_dir))


@st.cache_resource
def _load_texts(sample_size: int, data_file: str):
    cfg = load_config(sample_size=sample_size, data_file=data_file)
    reviews = load_benchmark(cfg)
    return dict(zip(reviews.ids, reviews.raw_texts))


def _record(action: str, **kwargs) -> None:
    rec = FeedbackRecord(
        run_name=st.session_state["run_name"],
        reviewer=st.session_state["reviewer"],
        action=action,
        **kwargs,
    )
    append_record(st.session_state["session_file"], rec)
    st.toast(f"recorded: {action}")


def main() -> None:
    cfg = load_config()
    runs = sorted(
        (d for d in cfg.runs_dir.glob("*") if d.is_dir() and run_is_complete(d)),
        key=lambda d: d.name,
    )
    if not runs:
        st.error(f"No completed runs in {cfg.runs_dir}. Run a pipeline first.")
        return

    with st.sidebar:
        st.header("Review session")
        reviewer = st.text_input("Reviewer name", value="")
        run_dir = st.selectbox("Run", runs, format_func=lambda d: d.name)
        if not reviewer:
            st.warning("Enter your name — every record is attributed.")
            st.stop()

    art = _load_run(str(run_dir))
    st.session_state["reviewer"] = reviewer
    st.session_state["run_name"] = art.run_name
    if (
        "session_file" not in st.session_state
        or st.session_state.get("session_run") != art.run_name
    ):
        st.session_state["session_file"] = session_file(cfg.feedback_dir, art.run_name)
        st.session_state["session_run"] = art.run_name

    texts = _load_texts(
        art.manifest.get("sample_size", len(art.doc_ids)),
        art.manifest.get("data_file", "sample_hotels_5k.jsonl"),
    )

    st.title(f"Run: {art.run_name}")
    m = art.metrics
    cols = st.columns(6)
    for col, (name, key) in zip(cols, [
        ("clusters", "n_clusters"), ("noise", "noise_ratio"),
        ("silhouette", "silhouette"), ("C_v", "coherence_cv"),
        ("entropy", "rating_entropy"), ("runtime s", "runtime_s"),
    ]):
        col.metric(name, m.get(key) if m.get(key) is not None else "—")
    for flag in m.get("failure_flags", []):
        st.warning(flag)

    left, right = st.columns([1, 1])

    # ── 2-D scatter ───────────────────────────────────────────────────────
    with left:
        import plotly.express as px

        labels_str = [str(l) if l != -1 else "noise" for l in art.labels]
        fig = px.scatter(
            x=art.coords_2d[:, 0], y=art.coords_2d[:, 1],
            color=labels_str,
            hover_name=[texts.get(d, "")[:120] for d in art.doc_ids],
            opacity=0.6,
        )
        fig.update_traces(marker=dict(size=4))
        fig.update_layout(height=600, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Sign-off")
        st.caption(
            "Checking this records: “a human reviewed the clusters of the "
            "winning pipeline and confirmed they are thematically coherent.”"
        )
        note = st.text_input("Confirmation note (optional)")
        if st.button("Confirm run: clusters are thematically coherent"):
            _record("confirm_run", note=note or None)

        st.subheader("Reassign a document")
        doc_id = st.text_input("doc_id")
        target = st.number_input("target cluster id (-1 = noise)", value=-1, step=1)
        if st.button("Reassign") and doc_id:
            _record("reassign_doc", doc_id=doc_id, target_cluster_id=int(target))

    # ── Cluster list ──────────────────────────────────────────────────────
    with right:
        st.subheader(f"Clusters ({len(art.clusters)})")
        other_ids = art.cluster_ids
        for cid in other_ids:
            info = art.clusters[cid]
            terms = ", ".join(w for w, _ in (tuple(t) for t in info.top_terms[:8]))
            stars = f" · {info.mean_stars}★" if info.mean_stars is not None else ""
            with st.expander(
                f"**{cid} — {info.label}** ({info.size} docs{stars}) · {info.label_source}"
            ):
                st.caption(f"Top terms: {terms}")
                if info.summary:
                    st.write(info.summary)
                st.markdown("**Random samples** (not centroid-picked):")
                for d in info.sample_doc_ids:
                    st.markdown(f"- {texts.get(d, '(missing)')[:300]}")

                c1, c2, c3 = st.columns(3)
                new_label = c1.text_input("Label", value=info.label, key=f"lbl{cid}")
                if c1.button("Approve / rename", key=f"app{cid}"):
                    if new_label != info.label:
                        _record("rename_label", cluster_id=cid, new_label=new_label)
                    else:
                        _record("approve_label", cluster_id=cid)

                merge_into = c2.selectbox(
                    "Merge into", [c for c in other_ids if c != cid],
                    key=f"mrg{cid}",
                )
                if c2.button("Merge", key=f"mrgbtn{cid}"):
                    _record("merge_clusters", cluster_id=cid, merge_into=int(merge_into))

                if c3.button("Split (flag for re-clustering)", key=f"spl{cid}"):
                    _record("split_cluster", cluster_id=cid)
                if c3.button("Mark as junk", key=f"jnk{cid}"):
                    _record("mark_junk", cluster_id=cid)

    st.sidebar.markdown("---")
    st.sidebar.caption(
        f"Feedback file: `{st.session_state['session_file'].name}`\n\n"
        "Apply on next run: `python -m reviewscope_ml.hitl.apply_feedback "
        f"{art.run_name}`"
    )


main()
