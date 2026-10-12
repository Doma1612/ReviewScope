from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass(frozen=True)
class CorpusProfile:

    slug: str
    unit_noun: str
    domain_description: str
    has_ratings: bool
    instructions: dict[str, Optional[str]]
    label_prompt_variant: str = "neutral"
    # Apply corpus stop words only to keywords; embeddings need natural text.
    boilerplate_terms: tuple[str, ...] = field(default_factory=tuple)


HOTELS = CorpusProfile(
    slug="hotels",
    unit_noun="review",
    domain_description="customer reviews of hotels",
    has_ratings=True,
    # Frozen instruction texts underpin the results in docs/technology-selection.md.
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
    label_prompt_variant="neutral",
    boilerplate_terms=(
        "mr", "mrs", "madam", "president", "commissioner", "rapporteur",
        "parliament", "commission", "council", "house", "member", "members",
        "honourable", "colleagues", "debate", "report", "vote", "european",
        "union", "eu",
    ),
)

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
    return PROFILES.get(corpus_slug, NEUTRAL)
