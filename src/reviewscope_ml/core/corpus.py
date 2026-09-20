"""
Corpus profiles — the per-corpus knobs that used to be module constants.

The pipeline stages are corpus-agnostic; a handful of *texts* are not. An
embedding instruction that says "Represent the hotel review…" and a label
prompt that says "You are analyzing customer reviews" are not neutral
scaffolding — fed a parliamentary debate they actively mislead the model. This
module is the one place those corpus-facing strings live, so adding a corpus is
a registry entry rather than a hunt through the stages.

What a profile does NOT carry: anything the pipeline can measure for itself.
Cluster sizing scales from the unit count (``cluster.scaled_min_cluster_size``),
and whether ratings exist is read off the data (``ReviewSet.has_ratings``) —
the ``has_ratings`` flag here is documentation of intent for the corpus, used
to decide what to *report*, never to override what was actually loaded.

Cache safety
------------
Instruction slugs are baked into embedding cache filenames, but the corpus is
already part of that key (``pipelines.runner._corpus_prefix``), so two corpora
may reuse the same slug name with different text without colliding. The rule
that matters is the other one: **never change the text behind an existing
(corpus, slug) pair** — the cache would silently serve vectors built from the
old instruction. Add a new slug instead.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass(frozen=True)
class CorpusProfile:
    """Corpus-facing configuration for one dataset."""

    slug: str
    # Singular noun for the clustered document, used in generated prose and
    # report headers ("5,000 speeches" rather than "5,000 reviews").
    unit_noun: str
    # How a prompt should describe the corpus, e.g. "customer reviews of
    # hotels". Substituted into the domain-aware label prompt.
    domain_description: str
    # Whether this corpus carries a numeric rating. False disables Tier-3
    # rating entropy *reporting*; the loader independently reports what it
    # actually found, and the two must agree.
    has_ratings: bool
    # Embedding-instruction texts for this corpus, keyed by slug. Slug names
    # are shared across corpora (the sweep registry names them); the texts are
    # not. "no_inst" must always be present and map to None.
    instructions: dict[str, Optional[str]]
    # Which entry in ``label.ollama.PROMPT_VARIANTS`` labels this corpus.
    label_prompt_variant: str = "neutral"
    # Terms so ubiquitous in this corpus that they describe every cluster and
    # therefore distinguish none — added to the stop list for keyword
    # extraction only. Never removed before embedding: the embedder is trained
    # on natural text and needs the connective tissue.
    boilerplate_terms: tuple[str, ...] = field(default_factory=tuple)


HOTELS = CorpusProfile(
    slug="hotels",
    unit_noun="review",
    domain_description="customer reviews of hotels",
    has_ratings=True,
    # Notebook 04's instruction texts, verbatim. Frozen: docs/technology-
    # selection.md's numbers were produced under exactly these strings.
    instructions={
        "no_inst": None,
        "generic": "Represent the hotel review for topic clustering:",
        "domain": (
            "Represent the hotel review for clustering by theme "
            "(room quality, staff, location, breakfast, cleanliness, value):"
        ),
        "sentiment": "Represent the hotel review to capture the main sentiment and opinion:",
    },
    label_prompt_variant="v2_mention",
)

EUROPARL_EN = CorpusProfile(
    slug="europarl-en",
    unit_noun="speech",
    domain_description="speeches in the European Parliament",
    has_ratings=False,
    # Same slug roles as the hotel table so the sweep registry's per-candidate
    # instruction declarations carry over unchanged; only the text differs.
    # The "domain" themes are the Parliament's standing policy areas rather
    # than a guess at what the clusters will be.
    instructions={
        "no_inst": None,
        "generic": "Represent the parliamentary speech for topic clustering:",
        "domain": (
            "Represent the parliamentary speech for clustering by policy theme "
            "(economy and budget, environment and energy, foreign affairs, "
            "justice and civil liberties, agriculture and fisheries, "
            "institutional procedure):"
        ),
        "sentiment": (
            "Represent the parliamentary speech to capture the speaker's "
            "position and stance:"
        ),
    },
    # The neutral template, with domain_description above supplying its
    # {domain}. A parliament-specific template would duplicate v2's structure
    # to change one noun phrase.
    label_prompt_variant="neutral",
    # Every speech is addressed to the chair and refers to the institutions;
    # "president", "commission" and "parliament" would otherwise top the
    # keyword list of every single cluster.
    boilerplate_terms=(
        "mr", "mrs", "madam", "president", "commissioner", "rapporteur",
        "parliament", "commission", "council", "house", "member", "members",
        "honourable", "colleagues", "debate", "report", "vote", "european",
        "union", "eu",
    ),
)

# Fallback for any corpus without a registered profile — an uploaded CSV of
# support tickets, say. Deliberately says nothing about a domain: a wrong
# domain hint is worse than none.
NEUTRAL = CorpusProfile(
    slug="",
    unit_noun="document",
    domain_description="documents",
    has_ratings=False,
    instructions={
        "no_inst": None,
        "generic": "Represent the document for topic clustering:",
        "domain": "Represent the document for clustering by theme:",
        "sentiment": "Represent the document to capture its main sentiment and opinion:",
    },
    label_prompt_variant="neutral",
)

PROFILES: dict[str, CorpusProfile] = {p.slug: p for p in (HOTELS, EUROPARL_EN)}


def profile_for(corpus_slug: str) -> CorpusProfile:
    """Return the profile for *corpus_slug*, or :data:`NEUTRAL` if unregistered.

    Unregistered is the normal case for application uploads — every project
    gets its own ``proj-<id>`` slug — so this must never raise.
    """
    return PROFILES.get(corpus_slug, NEUTRAL)
