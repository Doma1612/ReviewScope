import uuid
from typing import Literal

from pydantic import BaseModel, Field


class ProjectChatMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str


class ProjectChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=4000)
    history: list[ProjectChatMessage] = Field(default_factory=list, max_length=8)


class ProjectChatSource(BaseModel):
    citation_id: str
    unit: Literal["document", "segment", "cluster"]
    document_id: uuid.UUID | None = None
    segment_id: uuid.UUID | None = None
    primary_key_value: str | None = None
    cluster_id: uuid.UUID
    cluster_label: str
    rating: float | None = None
    snippet: str
    parent_text: str | None = None
    distance: float | None = None
    size: int | None = None
    n_mentions: int | None = None
    sentiment_avg: float | None = None
    mean_stars: float | None = None


class ProjectChatResponse(BaseModel):
    message: ProjectChatMessage
    model: str
    sources: list[ProjectChatSource]
