# Dataset Selection

**Project:** ReviewScope  
**Status:** Partially decided — development order confirmed, see section headers.

---

## Context

Choosing the right datasets for development is not just a convenience decision — it shapes what the pipeline gets optimized for and what the demos can actually show. Three things should guide the selection:

1. **Pipeline coverage.** We need datasets that stress-test the full stack: short noisy text and long structured text, single-domain and multi-domain corpora, enough volume to make clustering meaningful.
2. **DATS relevance.** If we want the academic / discourse analysis framing to hold up, at least some datasets should reflect the kind of data social scientists actually work with.
3. **Generalization.** A pipeline that only works on one domain or text length is not a general-purpose platform. The dataset selection should force us to prove the approach holds across different text types and domains.

All datasets listed here are openly available. None require a data sharing agreement beyond free registration where noted.

---

## Proposed Datasets

### 1. Yelp Open Dataset ✓ MVP Dataset
**Initial development dataset — short-form user-generated text**

6.9 million reviews across restaurants, shops, and local services, with star ratings, timestamps, and business metadata included. The volume is large enough to stress-test the pipeline properly, and the metadata allows filtering by business category, city, or rating band — which makes for more controlled experiments than throwing the full corpus at the clusterer at once.

User-generated review text is a well-understood benchmark domain for NLP. It is short, noisy, opinionated, and high-volume — all properties that make it a useful baseline for evaluating embedding quality and cluster coherence.

- **Source:** [yelp.com/dataset](https://www.yelp.com/dataset) — free, requires registration
- **Size:** ~6.9M reviews, JSON
- **Text length:** Short to medium
- **Languages:** Primarily English

**Open question for the team:** The full dataset is large. We should work with a filtered subset (e.g. one city, one business category) during early development to keep iteration fast?

---

### 2. Amazon Customer Reviews
**Secondary dataset — cross-domain generalization**

A different domain of short-form user-generated text, covering product feedback across many categories. Available directly via HuggingFace with per-category filtering, so we can start narrow and broaden as the pipeline matures.

The value here is generalization: if the pipeline clusters Yelp reviews well, does it generalize to a structurally similar but topically different corpus without reconfiguration? Amazon Reviews is a clean test for that.

- **Source:** HuggingFace
- **Size:** Per-category subsets range from ~50k to several million reviews
- **Text length:** Short to medium
- **Languages:** Primarily English

**Open question for the team:** Which product category makes for the most representative demo? Electronics reviews tend to have detailed complaints; clothing reviews tend to be shorter and noisier.

---

### 3. EuroParl Corpus ✓ Phase 2
**Academic / discourse analysis dataset — DATS angle**

Parliamentary debate transcripts from the European Parliament, covering 21 languages. This is the kind of primary source material that social scientists and discourse analysts actually work with — long documents, formal argumentation, a clear temporal dimension, and genuine thematic variety across political topics.

This dataset directly supports the DATS integration story. DATS appears to be built for this kind of corpus. Showing ReviewScope's clustering and summarization pipeline applied to EuroParl debates makes the academic credibility of the project concrete.

The multilingual dimension is also useful — it forces us to think early about whether embeddings generalize across languages or whether we need language-specific models.

- **Source:** [statmt.org/europarl](https://www.statmt.org/europarl/) or HuggingFace — `Helsinki-NLP/europarl`
- **Size:** ~60M words per language pair
- **Text length:** Long (speeches, debate segments)
- **Languages:** 21 EU languages

**Open question for the team:** Do we work with a single language (English or German) first, or use the multilingual nature of the corpus as a feature from the start? The latter would require evaluating multilingual embedding models earlier than planned.

---

### 4. All the News (Kaggle)
**Discourse analysis dataset — long-form text and concept-over-time angle**

Roughly 200,000 news articles from major US outlets, covering 2016–2017 with publication dates, authors, and source metadata. The temporal metadata is the key asset here — it enables concept-over-time analysis (how does cluster X grow or shrink over the coverage period?), which is one of DATS's own features and would make for a compelling demo.

Long-form text also tests the summarization pipeline differently than short reviews. A cluster summary drawn from news articles needs to synthesize more complex content and handle a wider range of vocabulary.

- **Source:** Kaggle — "All the News" dataset (multiple versions available)
- **Size:** ~200k articles
- **Text length:** Long (news articles)
- **Languages:** English

**Open question for the team:** Is concept-over-time visualization in scope for the course project, or a later milestone? The answer affects how early we need this dataset.

---

## Summary

| Dataset | Text length | Domain | Generalization value | DATS relevance |
|---|---|---|---|---|
| Yelp Open Dataset | Short | Local services / hospitality | Baseline | Low |
| Amazon Reviews | Short–medium | E-commerce / products | Cross-domain check | Low |
| EuroParl | Long | Political discourse | Long-form + multilingual | High |
| All the News | Long | Media / news | Long-form + temporal | High |

The spread across short and long text, user-generated and formal language, and English-only vs. multilingual ensures the pipeline is not accidentally tuned to one data shape.

---

## One to Keep in Reserve

**Google Play Store Reviews** (Kaggle, multiple versions) — app reviews are short, often multilingual, and structurally different enough from Yelp and Amazon to serve as an additional generalization test. Not needed immediately but useful to have on the list.

---

## Development Order ✓ Decided

1. **Yelp** — MVP. Short-form, well-understood domain, enough volume to stress-test the pipeline without needing to handle multilingual or long-form complexity first.
2. **EuroParl** — Phase 2, once the MVP pipeline is stable. Long-form, formal language, multilingual, strong DATS relevance.
3. **Amazon Reviews** — cross-domain generalization check once the pipeline is proven on Yelp.
4. **All the News** — when concept-over-time visualization is on the roadmap.

---

## Open Questions

1. **Yelp subset:** Work with a filtered subset (one city or business category) during early development, or load a larger slice from the start?
2. **EuroParl language scope:** Start with a single language (English or German) or treat multilingual capability as a feature from the beginning? The latter requires evaluating multilingual embedding models earlier.
3. **German-language data:** Given the DATS / UHH angle, should at least one German-language corpus be prioritized to demonstrate multilingual capability?
