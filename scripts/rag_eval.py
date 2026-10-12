from __future__ import annotations

import argparse
import csv
import io
import json
import math
import random
import re
import subprocess
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy import sparse
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, CountVectorizer

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "data" / "cache" / "rag_eval"
OUT = ROOT / "data" / "runs" / "rag_eval"

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
CHAT_MODEL = "llama3.2:latest"  # OLLAMA_CHAT_MODEL of the api service
JUDGE_MODEL = "gemma3:4b"       # a different family than the generator

CANDIDATES, PER_DOC_CAP, K = 30, 2, 10

CORPORA = {
    "europarl": {
        "project_id": "08e2f047-2221-46f6-bdb4-d57978d49d3c",
        "unit": "European Parliament speech",
        "collection": "English European Parliament debates (1996-2011)",
        "stored_vectors": True,
        "queries": [
            "fish stocks and fishing quotas",
            "Turkey's accession to the European Union",
            "climate change and reducing CO2 emissions",
            "human rights situation in Russia",
            "agricultural spending in the EU budget",
            "BSE and mad cow disease",
            "asylum and immigration policy",
            "the financial crisis and banking regulation",
            "rights of airline passengers",
            "the Middle East peace process",
            "regulation of chemicals under REACH",
            "roaming charges for mobile phones",
            "regulation of social media platforms",
            "cryptocurrencies like Bitcoin",
        ],
    },
    "hotels": {
        "project_id": "1115f858-bcef-404f-b2b2-f46d7b531a1b",
        "unit": "hotel review",
        "collection": "Yelp reviews of hotels",
        "stored_vectors": False,
        "queries": [
            "noise at night from neighbours or the street",
            "rude or unhelpful front desk staff",
            "quality of the breakfast buffet",
            "dirty rooms, stains or bad smells",
            "problems with the shower or bathroom",
            "parking fees and availability",
            "walking distance to downtown attractions",
            "comfort of the beds and pillows",
            "the pool area",
            "slow or unreliable WiFi",
            "long waits at check-in",
            "hidden fees or resort fees on the bill",
            "charging stations for electric cars",
            "paying with cryptocurrency",
        ],
    },
}
NICHE = 2


@dataclass
class Corpus:
    seg_ids: np.ndarray
    doc_ids: np.ndarray
    pk: np.ndarray
    cluster_ids: np.ndarray
    cluster_labels: dict
    texts: list
    vectors: np.ndarray

    @property
    def clustered(self) -> np.ndarray:
        return self.cluster_ids != ""


def _psql_csv(sql: str) -> list[dict]:
    out = subprocess.run(
        ["docker", "exec", "reviewscope_db_1", "psql", "-U", "reviewscope", "-d", "reviewscope",
         "-c", f"\\copy ({sql}) to stdout with csv header"],
        check=True, capture_output=True, text=True,
    ).stdout
    csv.field_size_limit(sys.maxsize)
    return list(csv.DictReader(io.StringIO(out)))


_embedders: dict[str, object] = {}


def _embed(texts: list[str], device: str) -> np.ndarray:
    if device not in _embedders:
        from reviewscope_ml.embed.models import encode_settings
        from reviewscope_ml.embed.sentence_transformer import SentenceTransformerEmbedder

        batch_size, max_seq = encode_settings(EMBEDDING_MODEL, 256)
        _embedders[device] = SentenceTransformerEmbedder(
            EMBEDDING_MODEL, instruction="no_inst", device=device,
            batch_size=batch_size, show_progress=False, max_seq=max_seq,
        )
    return np.asarray(_embedders[device].encode(texts), dtype=np.float32)


def _normalise(v: np.ndarray) -> np.ndarray:
    return v / np.linalg.norm(v, axis=1, keepdims=True).clip(1e-12)


def load_corpus(name: str, device: str) -> tuple[Corpus, dict]:
    cfg = CORPORA[name]
    pid = cfg["project_id"]
    CACHE.mkdir(parents=True, exist_ok=True)
    path = CACHE / f"{name}.npz"
    info: dict = {}
    if not path.exists():
        vec_col = ", s.vector_pg::text as vec" if cfg["stored_vectors"] else ""
        rows = _psql_csv(
            "select s.id, s.document_id, d.primary_key_value as pk, coalesce(s.cluster_id::text, '') as cid, "
            f"s.text{vec_col} from segments s join documents d on d.id = s.document_id "
            f"where s.project_id = '{pid}' order by s.id"
        )
        texts = [r["text"] for r in rows]
        if cfg["stored_vectors"]:
            vectors = np.stack([np.fromstring(r["vec"][1:-1], sep=",", dtype=np.float32) for r in rows])
        else:
            vectors = _embed(texts, device)
        np.savez(
            path, seg_ids=np.array([r["id"] for r in rows]), doc_ids=np.array([r["document_id"] for r in rows]),
            pk=np.array([r["pk"] for r in rows]), cluster_ids=np.array([r["cid"] for r in rows]),
            texts=np.array(texts, dtype=object), vectors=vectors,
        )
    z = np.load(path, allow_pickle=True)
    labels = {r["id"]: r["label"] for r in _psql_csv(f"select id, label from clusters where project_id = '{pid}'")}
    corpus = Corpus(z["seg_ids"], z["doc_ids"], z["pk"], z["cluster_ids"], labels,
                    list(z["texts"]), _normalise(z["vectors"].astype(np.float32)))

    # Query and stored vectors must use the same embedding space.
    if cfg["stored_vectors"]:
        idx = np.random.default_rng(0).choice(len(corpus.texts), 200, replace=False)
        fresh = _normalise(_embed([corpus.texts[i] for i in idx], device))
        info["stored_vs_reencoded_cosine_min"] = float((fresh * corpus.vectors[idx]).sum(1).min())
    return corpus, info


class BM25:
    def __init__(self, texts: list[str], k1: float = 1.2, b: float = 0.75):
        self.vec = CountVectorizer(lowercase=True, stop_words="english", token_pattern=r"(?u)\b\w\w+\b")
        tf = self.vec.fit_transform(texts).tocsr().astype(np.float32)
        dl = np.asarray(tf.sum(1)).ravel()
        n = tf.shape[0]
        df = np.bincount(tf.indices, minlength=tf.shape[1])
        idf = np.log(1 + (n - df + 0.5) / (df + 0.5)).astype(np.float32)
        norm = k1 * (1 - b + b * dl / dl.mean())
        rows = np.repeat(np.arange(n), np.diff(tf.indptr))
        data = tf.data * (k1 + 1) / (tf.data + norm[rows]) * idf[tf.indices]
        self.w = sparse.csr_matrix((data, tf.indices, tf.indptr), shape=tf.shape).tocsc()
        self.vocab = self.vec.vocabulary_

    def scores(self, query: str) -> np.ndarray:
        cols = [self.vocab[t] for t in self.vec.build_analyzer()(query) if t in self.vocab]
        if not cols:
            return np.zeros(self.w.shape[0], dtype=np.float32)
        return np.asarray(self.w[:, cols].sum(1)).ravel()


def _policy(order: np.ndarray, doc_ids: np.ndarray) -> list[int]:
    out, per_doc = [], {}
    for i in order[:CANDIDATES]:
        d = doc_ids[i]
        if per_doc.get(d, 0) >= PER_DOC_CAP:
            continue
        per_doc[d] = per_doc.get(d, 0) + 1
        out.append(int(i))
        if len(out) == K:
            break
    return out


def _top(scores: np.ndarray, n: int) -> np.ndarray:
    part = np.argpartition(-scores, n)[:n]
    return part[np.argsort(-scores[part], kind="stable")]


class Retrievers:
    def __init__(self, corpus: Corpus, device: str):
        self.c = corpus
        self.device = device
        self.bm25 = BM25(corpus.texts)
        self._noise_penalty = np.where(corpus.clustered, 0.0, -np.inf).astype(np.float32)

    def run(self, queries: list[str]) -> dict[str, list[list[int]]]:
        qv = _normalise(_embed(queries, self.device))
        results = {"app": [], "dense_all": [], "bm25": [], "hybrid": []}
        for q, v in zip(queries, qv):
            dense = self.c.vectors @ v
            lex = self.bm25.scores(q)
            results["app"].append(_policy(_top(dense + self._noise_penalty, CANDIDATES), self.c.doc_ids))
            d_top, l_top = _top(dense, 200), _top(lex, 200)
            results["dense_all"].append(_policy(d_top, self.c.doc_ids))
            results["bm25"].append(_policy(l_top, self.c.doc_ids))
            fused: dict[int, float] = {}
            for ranking in (d_top, l_top):
                for r, i in enumerate(ranking):
                    fused[int(i)] = fused.get(int(i), 0.0) + 1.0 / (60 + r + 1)
            order = np.array(sorted(fused, key=lambda i: -fused[i]))
            results["hybrid"].append(_policy(order, self.c.doc_ids))
        return results


def ollama(base: str, model: str, messages: list[dict], options: dict | None = None, timeout: int = 300) -> str:
    body = {"model": model, "messages": messages, "stream": False}
    if options:
        body["options"] = options
    req = urllib.request.Request(f"{base}/api/chat", data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())["message"]["content"].strip()


def pmap(fn, items, workers: int = 4):
    with ThreadPoolExecutor(workers) as ex:
        return list(ex.map(fn, items))


def _content_tokens(text: str) -> set[str]:
    return {t for t in re.findall(r"[a-z]{3,}", text.lower()) if t not in ENGLISH_STOP_WORDS}


def known_item(corpus: Corpus, retr: Retrievers, cfg: dict, base: str, n: int, seed: int) -> dict:
    rng = random.Random(seed)
    eligible = [i for i, t in enumerate(corpus.texts) if len(t) >= 60]
    sample = rng.sample(eligible, n)

    def gen(i: int) -> str:
        prompt = (
            f"Here is one sentence from a {cfg['unit']}:\n\n\"{corpus.texts[i]}\"\n\n"
            "Write ONE natural question (at most 15 words) that a user of a search engine over "
            "many such texts might ask, and that this sentence directly answers. Be specific "
            "enough that this sentence is a good answer, but paraphrase: do not copy the "
            "sentence's wording. Output only the question."
        )
        q = ollama(base, JUDGE_MODEL, [{"role": "user", "content": prompt}], {"temperature": 0, "seed": seed})
        return q.strip().strip('"').splitlines()[0]

    queries = pmap(gen, sample)
    runs = retr.run(queries)
    records = []
    for qi, (i, q) in enumerate(zip(sample, queries)):
        src_tokens = _content_tokens(corpus.texts[i])
        q_tokens = _content_tokens(q)
        rec = {
            "segment_index": i, "query": q, "source": corpus.texts[i],
            "source_clustered": bool(corpus.clustered[i]),
            "query_overlap": len(q_tokens & src_tokens) / max(1, len(q_tokens)),
        }
        for name, res in runs.items():
            rank = res[qi].index(i) + 1 if i in res[qi] else None
            rec[f"{name}_rank"] = rank
            rec[f"{name}_dochit"] = any(corpus.doc_ids[j] == corpus.doc_ids[i] for j in res[qi])
        records.append(rec)

    def summarise(recs: list[dict]) -> dict:
        out = {"n": len(recs)}
        for name in runs:
            ranks = [r[f"{name}_rank"] for r in recs]
            out[name] = {
                "hit@1": np.mean([r == 1 for r in ranks]),
                "hit@5": np.mean([r is not None and r <= 5 for r in ranks]),
                "hit@10": np.mean([r is not None for r in ranks]),
                "mrr@10": np.mean([1 / r if r else 0 for r in ranks]),
                "ndcg@10": np.mean([1 / math.log2(r + 1) if r else 0 for r in ranks]),
                "doc_hit@10": np.mean([r[f"{name}_dochit"] for r in recs]),
            }
        return out

    return {
        "records": records,
        "all": summarise(records),
        "clustered_sources": summarise([r for r in records if r["source_clustered"]]),
        "noise_sources": summarise([r for r in records if not r["source_clustered"]]),
        "mean_query_overlap": float(np.mean([r["query_overlap"] for r in records])),
    }


def judge_relevance(base: str, cfg: dict, query: str, passage: str) -> int:
    prompt = (
        f"You assess search results over {cfg['collection']}.\n\n"
        f"Search query: {query}\n\nPassage: {passage}\n\n"
        "How relevant is the passage to the search query?\n"
        "2 = the passage is about the query topic and says something substantive about it\n"
        "1 = the passage touches the query topic only partially or in passing\n"
        "0 = the passage is not about the query topic\n"
        "Answer with the single digit only."
    )
    out = ollama(base, JUDGE_MODEL, [{"role": "user", "content": prompt}], {"temperature": 0, "num_predict": 4})
    m = re.search(r"[012]", out)
    return int(m.group()) if m else 0


def graded(corpus: Corpus, retr: Retrievers, cfg: dict, base: str) -> dict:
    queries = cfg["queries"]
    runs = retr.run(queries)
    pool = sorted({(qi, i) for res in runs.values() for qi, lst in enumerate(res) for i in lst})
    grades = dict(zip(pool, pmap(lambda p: judge_relevance(base, cfg, queries[p[0]], corpus.texts[p[1]]), pool)))

    def per_query(name: str, qi: int) -> dict:
        g = [grades[(qi, i)] for i in runs[name][qi]]
        ideal = sorted((grades[p] for p in grades if p[0] == qi), reverse=True)[:K]
        dcg = sum((2 ** x - 1) / math.log2(r + 2) for r, x in enumerate(g))
        idcg = sum((2 ** x - 1) / math.log2(r + 2) for r, x in enumerate(ideal))
        first = next((r + 1 for r, x in enumerate(g) if x >= 1), None)
        return {
            "p@5": np.mean([x >= 1 for x in g[:5]]) if g else 0.0,
            "p@10": np.mean([x >= 1 for x in g]) if g else 0.0,
            "strict_p@10": np.mean([x == 2 for x in g]) if g else 0.0,
            "ndcg@10": dcg / idcg if idcg else 0.0,
            "mrr": 1 / first if first else 0.0,
            "grades": g,
        }

    table = {name: [per_query(name, qi) for qi in range(len(queries))] for name in runs}
    head = slice(0, len(queries) - NICHE)
    summary = {
        name: {m: float(np.mean([q[m] for q in rows[head]])) for m in ("p@5", "p@10", "strict_p@10", "ndcg@10", "mrr")}
        for name, rows in table.items()
    }
    judged = [
        {"query": queries[qi], "text": corpus.texts[i], "grade": grades[(qi, i)]}
        for (qi, i) in pool
    ]
    return {"queries": queries, "runs": runs, "per_query": table, "summary": summary, "judged": judged}


_RENDER = r"""
import json, sys, uuid
from app.project_chat_schemas import ProjectChatSource
from app.services.project_rag import rag_messages
payload = json.load(sys.stdin)
sources = [ProjectChatSource(**s) for s in payload["sources"]]
print(json.dumps(rag_messages(payload["question"], [], sources)))
"""


def render_messages(question: str, sources: list[dict]) -> list[dict]:
    out = subprocess.run(
        ["docker", "exec", "-i", "reviewscope_api_1", "python", "-c", _RENDER],
        input=json.dumps({"question": question, "sources": sources}),
        check=True, capture_output=True, text=True,
    ).stdout
    return json.loads(out)


_CITE = re.compile(r"\[(S\d+)\]")


def _sentences(answer: str) -> list[str]:
    parts = []
    for line in answer.splitlines():
        line = line.strip().lstrip("-*• ").strip()
        if not line:
            continue
        parts.extend(s.strip() for s in re.split(r"(?<=[.!?])\s+(?=[A-Z])", line))
    return [s for s in parts if len(s) >= 25 and not s.endswith(":")]


def grounding(corpus: Corpus, cfg: dict, base: str, graded_runs: dict) -> dict:
    queries = cfg["queries"]
    out = []
    for qi, q in enumerate(queries):
        idx = graded_runs["app"][qi]
        sources = [
            {"citation_id": f"S{n + 1}", "unit": "segment", "document_id": str(corpus.doc_ids[i]),
             "segment_id": str(corpus.seg_ids[i]), "primary_key_value": str(corpus.pk[i]),
             "cluster_id": str(corpus.cluster_ids[i]), "cluster_label": corpus.cluster_labels.get(corpus.cluster_ids[i], ""),
             "snippet": corpus.texts[i]}
            for n, i in enumerate(idx)
        ]
        messages = render_messages(q, sources)
        t0 = time.time()
        answer = ollama(base, CHAT_MODEL, messages, {"seed": 42})
        latency = time.time() - t0
        by_id = {s["citation_id"]: s["snippet"] for s in sources}
        sents = _sentences(answer)
        cited = [(s, _CITE.findall(s)) for s in sents]
        all_ids = _CITE.findall(answer)

        def support(item):
            sent, ids = item
            ids = [i for i in ids if i in by_id]
            if not ids:
                return None
            evidence = "\n".join(f"[{i}] {by_id[i]}" for i in ids)
            prompt = (
                f"Evidence:\n{evidence}\n\nStatement: {_CITE.sub('', sent).strip()}\n\n"
                "Is every factual claim in the statement supported by the evidence above? "
                "Paraphrase is fine; added facts, numbers or generalisations are not. Answer yes or no."
            )
            v = ollama(base, JUDGE_MODEL, [{"role": "user", "content": prompt}], {"temperature": 0, "num_predict": 4})
            return v.lower().startswith("y")

        verdicts = pmap(support, [c for c in cited if c[1]])
        judged = [v for v in verdicts if v is not None]
        out.append({
            "query": q, "niche": qi >= len(queries) - NICHE, "answer": answer, "latency_s": latency,
            "n_sentences": len(sents),
            "citation_validity": (sum(i in by_id for i in all_ids) / len(all_ids)) if all_ids else None,
            "citation_coverage": (sum(1 for _, ids in cited if ids) / len(sents)) if sents else None,
            "faithfulness": (sum(judged) / len(judged)) if judged else None,
            "abstains": bool(re.search(r"insufficient|not enough|no (?:relevant |direct )?(?:evidence|information|mention)|does not (?:mention|contain|provide)|(?:cannot|can't|couldn't|could not) (?:find|answer)|don't have", answer, re.I)),
            "sentence_verdicts": [{"sentence": c[0], "supported": v} for c, v in zip([c for c in cited if c[1]], verdicts)],
        })

    def mean(key, rows):
        vals = [r[key] for r in rows if r[key] is not None]
        return float(np.mean(vals)) if vals else None

    main = [r for r in out if not r["niche"]]
    return {
        "answers": out,
        "summary": {
            "citation_validity": mean("citation_validity", main),
            "citation_coverage": mean("citation_coverage", main),
            "faithfulness": mean("faithfulness", main),
            "answers_without_citation": sum(1 for r in main if r["citation_validity"] is None),
            "latency_s_median": float(np.median([r["latency_s"] for r in main])),
            "niche_abstained": [r["abstains"] for r in out if r["niche"]],
        },
    }


def _jsonable(o):
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, np.bool_):
        return bool(o)
    raise TypeError(type(o))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--corpus", choices=sorted(CORPORA), required=True)
    ap.add_argument("--ollama", default="http://127.0.0.1:11434")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--n-known-item", type=int, default=200)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    cfg = CORPORA[args.corpus]
    t0 = time.time()
    corpus, info = load_corpus(args.corpus, args.device)
    retr = Retrievers(corpus, args.device)
    print(f"[{args.corpus}] {len(corpus.texts):,} segments, {corpus.clustered.mean():.1%} clustered", flush=True)

    a = known_item(corpus, retr, cfg, args.ollama, args.n_known_item, args.seed)
    print(f"[{args.corpus}] A done ({time.time() - t0:.0f}s)", flush=True)
    b = graded(corpus, retr, cfg, args.ollama)
    print(f"[{args.corpus}] B done ({time.time() - t0:.0f}s)", flush=True)
    c = grounding(corpus, cfg, args.ollama, b["runs"])
    print(f"[{args.corpus}] C done ({time.time() - t0:.0f}s)", flush=True)

    OUT.mkdir(parents=True, exist_ok=True)
    result = {
        "corpus": args.corpus, "project_id": cfg["project_id"], "segments": len(corpus.texts),
        "clustered_share": float(corpus.clustered.mean()), "embedding_model": EMBEDDING_MODEL,
        "chat_model": CHAT_MODEL, "judge_model": JUDGE_MODEL, "seed": args.seed,
        "policy": {"candidates": CANDIDATES, "per_doc_cap": PER_DOC_CAP, "k": K},
        "info": info, "known_item": a, "graded": b, "grounding": c,
        "runtime_s": time.time() - t0,
    }
    path = OUT / f"rag_eval_{args.corpus}.json"
    path.write_text(json.dumps(result, indent=1, default=_jsonable))
    print(f"wrote {path}")
    print(json.dumps({"A": a["all"], "B": b["summary"], "C": c["summary"]}, indent=1, default=_jsonable))


if __name__ == "__main__":
    main()
