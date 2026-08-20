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


@st.cache_resource
def _candidate_clusters(sample_size: int, model: str, instruction: str, sentence_level: bool):
    """Rebuild one candidate's clustering from the sweep's caches.

    Cheap because everything is cached by the sweep — no model is loaded and
    nothing is re-embedded; this reads the same arrays the ranking was computed
    from, so what the reviewer inspects IS what the table scored.
    """
    from reviewscope_ml.eval.label_sweep import build_clusters
    from reviewscope_ml.represent.terms import ctfidf_terms

    cfg = load_config(sample_size=sample_size)
    units, embeddings, labels = build_clusters(cfg, model, instruction, sentence_level)
    terms = ctfidf_terms(units.texts, labels, top_n=10)
    return units, labels, terms


# Report families in data/runs, in the order the selection argument is made:
# what was admissible, how the embeddings ranked, how the labelers scored, and
# the sheet the humans fill in. Grouping them beats one flat glob because the
# directory also accumulates per-run subdirectories and raw logs.
_REPORT_KINDS = {
    "Admissibility (hard constraints + cost)": "admissibility_*.md",
    "Embedding sweep (ranking)": "model_sweep_*.md",
    "Labeler comparison (automatic)": "label_quality_*.md",
    "Human scoring sheet (static export)": "label_sheet_*.md",
}


def _report_browser(cfg, key: str) -> None:
    """Every generated report, browsable in the app that acts on them.

    These are the documents the technology-selection decision cites. Reading
    them next to the clusters they describe is the whole point — a reviewer who
    has to alt-tab to a text editor to see the metrics will score the labels
    without them.
    """
    found = {
        kind: sorted(cfg.runs_dir.glob(pattern))
        for kind, pattern in _REPORT_KINDS.items()
    }
    found = {k: v for k, v in found.items() if v}
    if not found:
        st.error(
            f"No reports in {cfg.runs_dir}. Generate them with "
            "`python -m reviewscope_ml.eval.model_sweep --sentence-level` and "
            "`python -m reviewscope_ml.eval.label_sweep`."
        )
        return

    kind = st.selectbox("Report family", list(found), key=f"{key}_kind")
    report = st.selectbox(
        "Report", found[kind], format_func=lambda p: p.name, key=f"{key}_report"
    )
    st.caption(f"`{report.relative_to(cfg.project_root)}`")
    st.markdown(report.read_text())


def _model_selection_view(cfg) -> None:
    """Pair the sweep's quantitative ranking with qualitative inspection.

    Tier 1 alone is not a decision (methodology §8): instruction-tuned models
    mechanically inflate silhouette by reshaping the space toward the
    instruction without moving coherence. The only way to catch that is to read
    the clusters a candidate actually produces, so the ranking table and the
    exemplar mentions live on the same page.
    """
    st.title("Embedding model selection")

    _report_browser(cfg, key="modelsel")

    st.markdown("---")
    st.subheader("Qualitative inspection")
    st.caption(
        "The ranking is a shortlist, not a verdict. Read the clusters: a model "
        "that wins Tier 1 while producing sentiment blobs ('Great Stay', "
        "'Terrible Experience') or one giant catch-all cluster has not won."
    )

    from reviewscope_ml.embed.models import CANDIDATES

    pairs = [(c.model, i) for c in CANDIDATES for i in c.instruction_variants()]
    choice = st.selectbox(
        "Candidate", pairs, format_func=lambda p: f"{p[0]}  ({p[1]})"
    )
    sample_size = st.number_input("Sample size", value=5_000, step=1_000, min_value=1_000)
    sentence_level = st.checkbox("Sentence/mention unit", value=True)

    if not st.button("Load clusters"):
        return
    try:
        units, labels, terms = _candidate_clusters(
            int(sample_size), choice[0], choice[1], sentence_level
        )
    except SystemExit as e:
        st.error(str(e))
        return
    except Exception as e:  # noqa: BLE001
        st.error(f"Could not rebuild this candidate's clusters: {e}")
        return

    cluster_ids = sorted(int(c) for c in set(labels.tolist()) if c != -1)
    noise = float((labels == -1).mean())
    unit_word = "mentions" if sentence_level else "reviews"
    st.write(
        f"**{len(cluster_ids)} clusters** over {len(units.texts):,} {unit_word} · "
        f"noise {noise:.1%}"
    )

    rng = np.random.default_rng(42)
    sizes = {cid: int((labels == cid).sum()) for cid in cluster_ids}
    top_n = st.slider("Clusters to show (largest first)", 3, 30, 10)
    for cid in sorted(cluster_ids, key=lambda c: -sizes[c])[:top_n]:
        idx = np.flatnonzero(labels == cid)
        share = sizes[cid] / len(labels)
        with st.expander(
            f"Cluster {cid} — {sizes[cid]:,} {unit_word} ({share:.1%}) — "
            + ", ".join(w for w, _ in terms.get(cid, [])[:5])
        ):
            st.caption("Top terms: " + ", ".join(w for w, _ in terms.get(cid, [])[:10]))
            # Random members, never centroid-nearest: centroid samples flatter
            # the cluster and hide exactly the fringe that reveals a bad model.
            pick = rng.choice(idx, size=min(6, len(idx)), replace=False)
            for i in pick:
                st.markdown(f"> {units.texts[i][:300]}")

    st.markdown("---")
    if st.button("Record this candidate as the reviewed choice"):
        _record(
            "confirm_model",
            note=f"{choice[0]} ({choice[1]}) reviewed qualitatively at "
                 f"{sample_size} {'segments' if sentence_level else 'documents'}",
        )


@st.cache_resource
def _labeler_clustering(sample_size: int, model: str, instruction: str):
    """The fixed clustering the labelers were judged on, with terms.

    Same cached arrays the labeler sweep used, so the clusters a reviewer reads
    here are byte-for-byte the ones the labels were generated for.
    """
    return _candidate_clusters(sample_size, model, instruction, True)


def _scoring_context(cfg, report: Path, sample_size: int, model: str, instruction: str):
    """Everything the scoring page needs: items, exemplars, terms, sizes."""
    from reviewscope_ml.eval.label_scoring import (
        build_scoring_items,
        load_label_records,
        select_exemplars,
    )

    records = load_label_records(report)
    items = build_scoring_items(records, seed=cfg.seed)
    units, labels, terms = _labeler_clustering(sample_size, model, instruction)

    sizes = {int(c): int((labels == c).sum()) for c in set(labels.tolist()) if c != -1}
    # The sheet's exemplar RNG is advanced once per cluster in size-descending
    # order, so the same order has to be reproduced here or the reviewer sees
    # different mentions than the exported sheet shows.
    chosen = sorted(
        (cid for cid in sizes if cid in items), key=lambda c: -sizes[c]
    )
    exemplars = select_exemplars(labels, units.texts, chosen, cfg.seed, k=5)
    return items, chosen, sizes, terms, exemplars, units, labels


def _labeler_scoring_view(cfg) -> None:
    """Blind human scoring of candidate labels — the qualitative half.

    `label_quality.md` ranks labelers on format compliance, discrimination and
    grounding. All three are proxies: a label can satisfy every one of them and
    still be useless to a reader ("Guest Experience Reviews" over a cluster
    about parking). Usefulness is the property the label exists for and the one
    no metric sees, so it is scored here by people and recorded in the same
    append-only trail as every other human decision.

    Blind by construction — the producing model is hidden while scoring, and
    the presentation order is shuffled per cluster. The exported markdown sheet
    could not do either: it prints the labeler's name in the row.
    """
    from reviewscope_ml.eval.label_scoring import (
        collect_scores,
        coverage,
        krippendorff_alpha,
        pairwise_agreement,
        score_by_labeler,
    )
    from reviewscope_ml.hitl.feedback import load_feedback
    from reviewscope_ml.pipelines.spec import SENTENCE_EMBEDDING

    st.title("Label scoring")

    reports = sorted(cfg.runs_dir.glob("label_quality_*.json"))
    if not reports:
        st.error(
            "No labeler results in data/runs. Run "
            "`python -m reviewscope_ml.eval.label_sweep --sample-size 5000` first."
        )
        return

    with st.sidebar:
        report = st.selectbox("Labeler run", reports, format_func=lambda p: p.name)
        st.caption("Clustering the labels were generated for:")
        sample_size = st.number_input(
            "Sample size", value=5_000, step=1_000, min_value=1_000
        )
        model = st.text_input("Embedding model", value=SENTENCE_EMBEDDING)
        instruction = st.text_input("Instruction", value="no_inst")

    # One scoring campaign per labeler run, so every reviewer's session file
    # aggregates into the same set and agreement can be computed across them.
    run_name = f"label_scoring__{report.stem}"
    st.session_state["run_name"] = run_name
    if st.session_state.get("session_run") != run_name:
        st.session_state["session_file"] = session_file(cfg.feedback_dir, run_name)
        st.session_state["session_run"] = run_name

    try:
        items, chosen, sizes, terms, exemplars, units, labels = _scoring_context(
            cfg, report, int(sample_size), model, instruction
        )
    except SystemExit as e:
        st.error(str(e))
        return
    except Exception as e:  # noqa: BLE001
        st.error(f"Could not rebuild the labeled clustering: {e}")
        return

    reviewer = st.session_state["reviewer"]
    scores = collect_scores(load_feedback(cfg.feedback_dir, run_name))
    mine, total = coverage(items, scores, reviewer=reviewer)
    anyone, _ = coverage(items, scores)

    c1, c2, c3 = st.columns(3)
    c1.metric("Your judgements", f"{mine}/{total}")
    c2.metric("Scored by anyone", f"{anyone}/{total}")
    c3.metric("Reviewers", len({r for per in scores.values() for r in per}))
    st.progress(mine / total if total else 0.0)

    tab_score, tab_results, tab_reports = st.tabs(
        ["Score", "Results & agreement", "Reports"]
    )

    with tab_score:
        _scoring_tab(
            items, chosen, sizes, terms, exemplars, units, labels, scores, reviewer
        )

    with tab_results:
        st.caption(
            "Human score is the mean 1-5 rating of the label a model produced, "
            "averaged over scored clusters. It is not comparable to the "
            "composite in `label_quality.md` — that one measures form, this one "
            "measures usefulness. Read them side by side; a model that wins one "
            "and loses the other is the interesting case."
        )
        rows = score_by_labeler(items, scores)
        if not rows:
            st.info("Nothing scored yet — start on the Score tab.")
        else:
            st.dataframe(rows, width="stretch")

        st.subheader("Reviewer agreement")
        st.caption(
            "One reviewer's sheet is one opinion. Agreement is what separates a "
            "shared standard from a shared blind spot, so it is reported even "
            "when it is bad — especially then."
        )
        alpha = krippendorff_alpha(scores)
        pairs = pairwise_agreement(scores)
        if alpha is None:
            st.info(
                "No label has been scored by two different reviewers yet, so "
                "agreement is undefined. Have a second person score the same "
                "run before citing any of these numbers."
            )
        else:
            st.metric("Krippendorff's α (ordinal)", f"{alpha:.3f}")
            st.caption(
                "≥0.80 conventionally reliable · 0.67-0.80 tentative · "
                "<0.67 the scale or the instructions need work, not the models."
            )
        if pairs:
            st.dataframe(pairs, width="stretch")

    with tab_reports:
        _report_browser(cfg, key="scoring")


def _scoring_tab(items, chosen, sizes, terms, exemplars, units, labels, scores, reviewer):
    """One cluster at a time: read the mentions, score every candidate label."""
    if not chosen:
        st.error(
            "The labeler run's cluster ids are not present in this clustering — "
            "check the sample size, embedding model and instruction in the "
            "sidebar against the header of the matching `label_quality_*.md`."
        )
        return

    done_ids = {
        cid for cid in chosen
        if all(scores.get((cid, i.key), {}).get(reviewer) for i in items[cid])
    }
    remaining = [cid for cid in chosen if cid not in done_ids]

    def fmt(cid: int) -> str:
        mark = "✓" if cid in done_ids else "○"
        return f"{mark} Cluster {cid} — {sizes[cid]:,} mentions"

    default = chosen.index(remaining[0]) if remaining else 0
    cid = st.selectbox("Cluster", chosen, index=default, format_func=fmt)

    st.subheader(f"Cluster {cid} — {sizes[cid]:,} mentions")
    st.caption("Top terms: " + ", ".join(w for w, _ in terms.get(cid, [])[:10]))
    st.markdown("**Exemplar mentions** (random members, not centroid-nearest):")
    for text in exemplars.get(cid, []):
        st.markdown(f"> {text}")

    with st.expander("Draw more members (use when the five above are unclear)"):
        # Five mentions are enough to score a good label and not always enough
        # to convict a bad one, so a reviewer can keep sampling. The draw is
        # seeded per press so it is reproducible from the record if questioned.
        draw = st.session_state.get(f"draw_{cid}", 0)
        if st.button("Draw 10 more", key=f"more_{cid}"):
            st.session_state[f"draw_{cid}"] = draw + 1
            st.rerun()
        if draw:
            idx = np.flatnonzero(labels == cid)
            rng = np.random.default_rng(1000 + draw)
            for i in rng.choice(idx, size=min(10, len(idx)), replace=False):
                st.markdown(f"> {units.texts[i][:300]}")

    st.markdown("---")
    st.markdown(
        f"**Score each label 1-5 against those mentions.** "
        f"1 = wrong or uselessly generic · 3 = correct but vague · "
        f"5 = specific and correct. "
        f"Producing model is hidden until you submit."
    )

    with st.form(key=f"score_form_{cid}"):
        chosen_scores: dict[str, tuple[str, int, str]] = {}
        for n, item in enumerate(items[cid], start=1):
            st.markdown(f"**{n}. {item.label}**")
            prior = scores.get((cid, item.key), {}).get(reviewer)
            cols = st.columns([3, 5])
            value = cols[0].radio(
                "score",
                [1, 2, 3, 4, 5],
                index=(prior - 1) if prior else None,
                horizontal=True,
                key=f"sc_{cid}_{n}",
                label_visibility="collapsed",
            )
            note = cols[1].text_input(
                "note", key=f"nt_{cid}_{n}", label_visibility="collapsed",
                placeholder="optional note — why this score",
            )
            if value:
                chosen_scores[item.key] = (item.label, int(value), note)
            st.markdown("")

        if st.form_submit_button("Save scores for this cluster"):
            if not chosen_scores:
                st.warning("Nothing scored.")
            else:
                for label, value, note in chosen_scores.values():
                    _record(
                        "score_label",
                        cluster_id=int(cid),
                        label=label,
                        score=value,
                        note=note or None,
                    )
                st.success(f"Saved {len(chosen_scores)} scores for cluster {cid}.")
                st.rerun()

    with st.expander("Reveal which model produced each label"):
        st.caption(
            "Kept behind a click so it cannot bias a score. Open it after "
            "scoring, or when you need to debug a labeler rather than judge it."
        )
        for n, item in enumerate(items[cid], start=1):
            who = ", ".join(f"`{m}` ({v})" for m, v in item.produced_by)
            st.markdown(f"{n}. **{item.label}** — {who}")


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
    st.button("← Back to overview", on_click=_back_to_overview)
    if not ids:
        st.warning("No (existing) clusters selected.")
        return

    st.title("Cluster detail view")
    for cid in ids:
        info = art.clusters[cid]
        senti = (
            f" · Sentiment {info.sentiment_avg:+.2f}"
            if info.sentiment_avg is not None else ""
        )
        docs = f" in {info.n_documents} Reviews" if info.n_documents is not None else ""
        st.markdown(
            f"**{cid} — {info.label}** · {info.size} units{docs}{senti} · "
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
            "Sentiment filter", ["negative", "neutral", "positive"], default=[]
        )
        if senti_filter:
            df = df[df["sentiment"].isin(senti_filter)]
    query = fcol2.text_input("Text search within these clusters")
    if query:
        df = df[df["text"].str.contains(query, case=False, na=False)]

    st.caption(f"{len(df):,} data points (sort by clicking the column headers)")
    st.dataframe(df, height=480, width="stretch", hide_index=True)

    # ── Aktionen ──────────────────────────────────────────────────────────
    st.subheader("Actions")
    if len(ids) > 1:
        mcol1, mcol2 = st.columns([2, 1])
        target = mcol1.selectbox(
            "Merge all selected clusters into:",
            ids,
            format_func=lambda c: f"{c} — {art.clusters[c].label}",
        )
        if mcol2.button(f"Merge {len(ids) - 1} → {target}"):
            for cid in ids:
                if cid != target:
                    _record("merge_clusters", cluster_id=cid, merge_into=int(target))
            st.info(
                "Merge recorded — takes effect on the next apply_feedback."
            )

    for cid in ids:
        info = art.clusters[cid]
        c1, c2, c3 = st.columns([3, 1, 1])
        new_label = c1.text_input(f"Label cluster {cid}", value=info.label, key=f"dlbl{cid}")
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

    with st.sidebar:
        st.header("Review session")
        reviewer = st.text_input("Reviewer name", value="")
        mode = st.radio(
            "Mode", ["Review a run", "Model selection", "Label scoring"]
        )
        if not reviewer:
            st.warning("Enter your name — every record is attributed.")
            st.stop()

    st.session_state["reviewer"] = reviewer

    if mode == "Label scoring":
        # Its own campaign name is set inside the view, keyed to the labeler
        # run being scored so several reviewers' sessions aggregate.
        _labeler_scoring_view(cfg)
        return

    if mode == "Model selection":
        # Technology selection is not tied to one pipeline run, but its sign-off
        # still belongs in the same append-only trail.
        st.session_state["run_name"] = "model_selection"
        if st.session_state.get("session_run") != "model_selection":
            st.session_state["session_file"] = session_file(
                cfg.feedback_dir, "model_selection"
            )
            st.session_state["session_run"] = "model_selection"
        _model_selection_view(cfg)
        return

    if not runs:
        st.error(f"No completed runs in {cfg.runs_dir}. Run a pipeline first.")
        return
    with st.sidebar:
        run_dir = st.selectbox("Run", runs, format_func=lambda d: d.name)

    art = _load_run(str(run_dir))
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
            "Focus clusters (empty = all)",
            options=art.cluster_ids,
            format_func=lambda c: cluster_names[c],
            key="focus_clusters",
        )
        st.button(
            f"🔍 Detail view for selection ({len(focus)} clusters)" if focus
            else "🔍 Detail view (select clusters first)",
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
            notes.append(f"showing {MAX_PLOT_POINTS:,} of {n:,} points (random sample)")
        if focus:
            notes.append(f"{int(shown.sum()):,} points in {len(focus)} focused clusters")
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
                    "🔍 Detail view (all data points + metadata)",
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
