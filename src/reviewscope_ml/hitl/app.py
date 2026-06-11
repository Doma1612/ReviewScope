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
def _load_unit_data(sample_size: int, data_file: str, unit: str):
    """id -> text and id -> star maps for the run's unit (docs or segments)."""
    cfg = load_config(sample_size=sample_size, data_file=data_file)
    reviews = load_benchmark(cfg)
    if unit == "sentence":
        # Segmentation is deterministic, so segment ids in the artifact
        # resolve against a re-derived segment set.
        from reviewscope_ml.data.segment import segment_reviews

        reviews = segment_reviews(reviews)
    texts = dict(zip(reviews.ids, reviews.raw_texts))
    stars = dict(zip(reviews.ids, (float(s) for s in reviews.stars)))
    return texts, stars


def _record(action: str, **kwargs) -> None:
    rec = FeedbackRecord(
        run_name=st.session_state["run_name"],
        reviewer=st.session_state["reviewer"],
        action=action,
        **kwargs,
    )
    append_record(st.session_state["session_file"], rec)
    st.toast(f"recorded: {action}")


def _open_detail(cluster_ids: list[int]) -> None:
    st.session_state["view"] = "detail"
    st.session_state["detail_ids"] = list(cluster_ids)


def _back_to_overview() -> None:
    st.session_state["view"] = "overview"


def _detail_view(art, texts: dict, stars: dict) -> None:
    """
    Drill-down: every data point of the selected cluster(s) with full
    metadata, plus the cluster actions (rename/approve/junk/merge) so a
    reviewer can investigate and act in one place. Like everything in this
    app, actions only append feedback records — artifacts change on
    apply_feedback, never live.
    """
    import pandas as pd

    ids = [c for c in st.session_state.get("detail_ids", []) if c in art.clusters]
    st.button("← Zurück zur Übersicht", on_click=_back_to_overview)
    if not ids:
        st.warning("Keine (existierenden) Cluster ausgewählt.")
        return

    st.title("Cluster-Detailansicht")
    for cid in ids:
        info = art.clusters[cid]
        senti = (
            f" · Sentiment {info.sentiment_avg:+.2f}"
            if info.sentiment_avg is not None else ""
        )
        docs = f" in {info.n_documents} Reviews" if info.n_documents is not None else ""
        st.markdown(
            f"**{cid} — {info.label}** · {info.size} Einträge{docs}{senti} · "
            f"Terms: {', '.join(w for w, _ in (tuple(t) for t in info.top_terms[:8]))}"
        )

    # ── Datentabelle: alle Punkte der ausgewählten Cluster ────────────────
    mask = np.isin(art.labels, ids)
    idxs = np.flatnonzero(mask)
    has_sentiment = art.sentiment_scores is not None
    df = pd.DataFrame({
        "cluster": [int(art.labels[i]) for i in idxs],
        "label": [art.clusters[int(art.labels[i])].label for i in idxs],
        "text": [texts.get(art.doc_ids[i], "") for i in idxs],
        "sentiment_score": (
            [round(float(art.sentiment_scores[i]), 3) for i in idxs]
            if has_sentiment else None
        ),
        "sentiment": (
            [art.sentiment_labels[i] for i in idxs] if has_sentiment else None
        ),
        "stars": [stars.get(art.doc_ids[i]) for i in idxs],
        "doc_id": [art.doc_ids[i] for i in idxs],
        **(
            {"micro_cluster": [int(art.micro_labels[i]) for i in idxs]}
            if art.micro_labels is not None else {}
        ),
    })

    fcol1, fcol2 = st.columns([1, 2])
    if has_sentiment:
        senti_filter = fcol1.multiselect(
            "Sentiment-Filter", ["negative", "neutral", "positive"], default=[]
        )
        if senti_filter:
            df = df[df["sentiment"].isin(senti_filter)]
    query = fcol2.text_input("Textsuche in diesen Clustern")
    if query:
        df = df[df["text"].str.contains(query, case=False, na=False)]

    st.caption(f"{len(df):,} Datenpunkte (sortierbar per Klick auf die Spaltenköpfe)")
    st.dataframe(df, height=480, width="stretch", hide_index=True)

    # ── Aktionen ──────────────────────────────────────────────────────────
    st.subheader("Aktionen")
    if len(ids) > 1:
        mcol1, mcol2 = st.columns([2, 1])
        target = mcol1.selectbox(
            "Alle ausgewählten Cluster mergen in:",
            ids,
            format_func=lambda c: f"{c} — {art.clusters[c].label}",
        )
        if mcol2.button(f"Merge {len(ids) - 1} → {target}"):
            for cid in ids:
                if cid != target:
                    _record("merge_clusters", cluster_id=cid, merge_into=int(target))
            st.info(
                "Merge aufgezeichnet — wird beim nächsten apply_feedback wirksam."
            )

    for cid in ids:
        info = art.clusters[cid]
        c1, c2, c3 = st.columns([3, 1, 1])
        new_label = c1.text_input(f"Label Cluster {cid}", value=info.label, key=f"dlbl{cid}")
        if c2.button("Approve / rename", key=f"dapp{cid}"):
            if new_label != info.label:
                _record("rename_label", cluster_id=cid, new_label=new_label)
            else:
                _record("approve_label", cluster_id=cid)
        if c3.button("Junk", key=f"djnk{cid}"):
            _record("mark_junk", cluster_id=cid)


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

    texts, stars = _load_unit_data(
        art.manifest.get("sample_size", len(art.doc_ids)),
        art.manifest.get("data_file", "sample_hotels_5k.jsonl"),
        art.manifest.get("unit", "document"),
    )

    if st.session_state.get("view") == "detail":
        _detail_view(art, texts, stars)
        return
    if art.manifest.get("unit") == "sentence":
        st.caption(
            "Sentence-level run: each point/sample is one **mention** (sentence); "
            "cluster sizes show mentions and distinct reviews."
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
    # Performance rules (sentence runs have 40k+ points and crashed browsers):
    # WebGL traces only (Scattergl, never SVG), exactly two traces regardless
    # of cluster count, a hard cap on displayed points, and hover payloads
    # built only for the points actually shown.
    MAX_PLOT_POINTS = 12_000

    with left:
        import plotly.express as px
        import plotly.graph_objects as go

        cluster_names = {
            cid: f"{cid} — {art.clusters[cid].label}" for cid in art.cluster_ids
        }
        focus = st.multiselect(
            "Cluster fokussieren (leer = alle)",
            options=art.cluster_ids,
            format_func=lambda c: cluster_names[c],
            key="focus_clusters",
        )
        st.button(
            f"🔍 Detailansicht für Auswahl ({len(focus)} Cluster)" if focus
            else "🔍 Detailansicht (erst Cluster auswählen)",
            disabled=not focus,
            on_click=_open_detail,
            args=(focus,),
        )

        n = len(art.doc_ids)
        if n > MAX_PLOT_POINTS:
            rng = np.random.default_rng(42)  # stable sample across reruns
            idx = np.sort(rng.choice(n, size=MAX_PLOT_POINTS, replace=False))
        else:
            idx = np.arange(n)
        xs, ys = art.coords_2d[idx, 0], art.coords_2d[idx, 1]
        point_labels = art.labels[idx]

        focus_set = set(focus)
        dimmed = (
            np.array([int(l) not in focus_set for l in point_labels])
            if focus else np.zeros(len(idx), dtype=bool)
        )

        palette = px.colors.qualitative.Alphabet
        fig = go.Figure()
        if dimmed.any():
            fig.add_trace(go.Scattergl(
                x=xs[dimmed], y=ys[dimmed], mode="markers",
                marker=dict(size=3, color="lightgrey", opacity=0.25),
                hoverinfo="skip", showlegend=False,
            ))
        shown = ~dimmed
        shown_idx = idx[shown]
        hover = []
        for i, l in zip(shown_idx, point_labels[shown]):
            doc_id = art.doc_ids[i]
            name = cluster_names.get(int(l), "noise")
            snippet = texts.get(doc_id, "")[:120]
            senti = (
                f" · {art.sentiment_labels[i]}"
                if art.sentiment_labels is not None else ""
            )
            hover.append(f"<b>{name}</b><br>{snippet}<br><i>{doc_id}</i>{senti}")
        fig.add_trace(go.Scattergl(
            x=xs[shown], y=ys[shown], mode="markers",
            marker=dict(
                size=5 if focus else 4,
                opacity=0.75,
                color=[
                    "lightgrey" if l == -1 else palette[int(l) % len(palette)]
                    for l in point_labels[shown]
                ],
            ),
            text=hover,
            hovertemplate="%{text}<extra></extra>",
            showlegend=False,
        ))
        fig.update_layout(height=600)
        st.plotly_chart(fig, width="stretch")

        notes = []
        if n > MAX_PLOT_POINTS:
            notes.append(f"Anzeige: {MAX_PLOT_POINTS:,} von {n:,} Punkten (Zufallsstichprobe)")
        if focus:
            notes.append(f"{int(shown.sum()):,} Punkte in {len(focus)} fokussierten Clustern")
        if notes:
            st.caption(" · ".join(notes))

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

    # ── Cluster list (follows the focus selection from the scatter) ──────
    with right:
        focused = st.session_state.get("focus_clusters") or []
        shown_ids = focused if focused else art.cluster_ids
        st.subheader(
            f"Clusters ({len(shown_ids)}/{len(art.clusters)})"
            if focused else f"Clusters ({len(art.clusters)})"
        )
        other_ids = art.cluster_ids
        for cid in shown_ids:
            info = art.clusters[cid]
            terms = ", ".join(w for w, _ in (tuple(t) for t in info.top_terms[:8]))
            stars_str = f" · {info.mean_stars}★" if info.mean_stars is not None else ""
            if info.n_documents is not None:
                count = f"{info.size} mentions in {info.n_documents} reviews"
            else:
                count = f"{info.size} docs"
            with st.expander(
                f"**{cid} — {info.label}** ({count}{stars_str}) · {info.label_source}"
            ):
                st.button(
                    "🔍 Detailansicht (alle Datenpunkte + Metadaten)",
                    key=f"det{cid}",
                    on_click=_open_detail,
                    args=([cid],),
                )
                st.caption(f"Top terms: {terms}")
                if info.sentiment_avg is not None:
                    d = info.sentiment_dist or {}
                    st.caption(
                        f"Sentiment: {info.sentiment_avg:+.2f} · "
                        f"😞 {d.get('negative', 0):.0%} / "
                        f"😐 {d.get('neutral', 0):.0%} / "
                        f"😊 {d.get('positive', 0):.0%}"
                    )
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
