from __future__ import annotations

import json
import re
import uuid

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import StreamingResponse
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.deps import get_current_user, require_project_role
from app.core.config import get_settings
from app.db.session import get_db
from app.models import Project, ProjectRole, User
from app.project_chat_schemas import ProjectChatRequest, ProjectChatResponse
from app.services.ollama_chat import OllamaChatError, chat_completion, stream_chat_completion
from app.services.project_chat_tools import plan_project_tool
from app.services.project_rag import count_project_clusters, rag_messages, rank_project_clusters, resolve_project_cluster, retrieve_project_sources, retrieve_related_clusters
from app.services.query_embeddings import embed_project_query


router = APIRouter()
settings = get_settings()
_GREETING = re.compile(r"^(?:hi|hello|hey)(?:\s+there)?[!,.\s]*$", re.IGNORECASE)

# Gate specialized tools on intent to avoid confidently answering an unasked question.
_TOOL_INTENT = {
    "rank_clusters": re.compile(
        r"\b(?:rank|ranking|largest|biggest|smallest|most|least|highest|lowest|top|bottom|worst|best)\b",
        re.IGNORECASE,
    ),
    "get_project_cluster_count": re.compile(r"\b(?:how many|number of|count)\b", re.IGNORECASE),
    "explain_project_data_source": re.compile(
        r"\b(?:how do you know|how did you|where .* from|data source|based on what)\b", re.IGNORECASE
    ),
    "describe_project_chat": re.compile(
        r"\b(?:what can you|who are you|help|capabilit)\b", re.IGNORECASE
    ),
}

_SEARCH_FALLBACK = ("search_project_evidence", {"query": None, "limit": 10})


def _is_greeting(message: str) -> bool:
    return bool(_GREETING.fullmatch(message.strip()))


def _search_fallback(question: str) -> tuple[str, dict]:
    name, arguments = _SEARCH_FALLBACK
    return name, {**arguments, "query": question}


async def _authorize_project_chat(db: AsyncSession, project_id: uuid.UUID, user: User) -> None:
    await require_project_role(db, project_id, user.id, {ProjectRole.owner, ProjectRole.viewer})


def _ranking_answer(sources, metric: str, direction: str) -> str:
    labels = {
        ("size", "asc"): "smallest",
        ("size", "desc"): "largest",
        ("sentiment", "asc"): "lowest sentiment",
        ("sentiment", "desc"): "highest sentiment",
        ("rating", "asc"): "lowest rated",
        ("rating", "desc"): "highest rated",
    }
    descriptor = labels[(metric, direction)]
    rows = []
    for source in sources:
        value = source.size if metric == "size" else source.sentiment_avg if metric == "sentiment" else source.mean_stars
        unit = "reviews" if metric == "size" else "sentiment score" if metric == "sentiment" else "average rating"
        rows.append(f"**{source.cluster_label}**: {value} {unit} [{source.citation_id}]")
    if len(rows) == 1:
        return f"The {descriptor} cluster is {rows[0]}."
    return f"The {descriptor} clusters are:\n" + "\n".join(f"- {row}" for row in rows)


def _related_clusters_answer(sources) -> str:
    return "Related clusters:\n" + "\n".join(
        f"- **{source.cluster_label}** [{source.citation_id}]" for source in sources
    )


async def _context(db: AsyncSession, project_id: uuid.UUID, user: User, body: ProjectChatRequest):
    await _authorize_project_chat(db, project_id, user)
    project = await db.get(Project, project_id)
    if not project:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Project not found")
    try:
        tool_name, arguments = await run_in_threadpool(
            plan_project_tool,
            base_url=settings.ollama_base_url,
            model=settings.ollama_chat_model,
            timeout=settings.ollama_timeout_seconds,
            question=body.message.strip(),
            history=body.history,
        )
    except OllamaChatError as exc:
        raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail=str(exc)) from exc
    except ValueError:
        tool_name, arguments = _search_fallback(body.message.strip())

    intent = _TOOL_INTENT.get(tool_name)
    if intent is not None and not intent.search(body.message):
        tool_name, arguments = _search_fallback(body.message.strip())

    if tool_name == "describe_project_chat":
        sources = []
        direct_answer = "I can find related clusters, explain why a named cluster relates to a topic, answer exact cluster counts and rankings, and retrieve review evidence for qualitative questions."
    elif tool_name == "explain_project_data_source":
        count = await count_project_clusters(db, project)
        sources = []
        direct_answer = f"I read the current cluster records stored for this project. Counting those records gives **{count}** clusters; unclustered noise is not counted as a cluster."
    elif tool_name == "get_project_cluster_count":
        count = await count_project_clusters(db, project)
        sources = []
        direct_answer = f"This project has **{count}** clusters."
    elif tool_name == "rank_clusters":
        sources = await rank_project_clusters(db, project, **arguments)
        direct_answer = _ranking_answer(sources, arguments["metric"], arguments["direction"])
    elif tool_name == "search_related_clusters":
        if not project.retrieval_ready:
            raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="Semantic search is unavailable until this project is reprocessed.")
        try:
            vector = await run_in_threadpool(embed_project_query, project, arguments["query"])
        except ValueError as exc:
            raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(exc)) from exc
        sources = await retrieve_related_clusters(db, project, vector, **arguments)
        direct_answer = _related_clusters_answer(sources)
    elif tool_name == "explain_cluster_relation":
        if not project.retrieval_ready:
            raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="Semantic search is unavailable until this project is reprocessed.")
        cluster = await resolve_project_cluster(db, project, arguments["cluster_reference"])
        if not cluster:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="The named cluster was not found in this project.")
        try:
            vector = await run_in_threadpool(embed_project_query, project, arguments["topic"])
        except ValueError as exc:
            raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(exc)) from exc
        sources = await retrieve_project_sources(db, project, vector, limit=arguments["limit"], cluster_id=cluster.id)
        direct_answer = None
    else:
        if not project.retrieval_ready:
            raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="Semantic search is unavailable until this project is reprocessed.")
        try:
            vector = await run_in_threadpool(embed_project_query, project, arguments["query"])
        except ValueError as exc:
            raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(exc)) from exc
        sources = await retrieve_project_sources(db, project, vector, limit=arguments["limit"])
        direct_answer = None
    if not sources and direct_answer is None:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="The selected tool returned no project evidence.")
    return sources, rag_messages(body.message.strip(), body.history, sources), direct_answer


@router.post("/{project_id}/chat", response_model=ProjectChatResponse)
async def project_chat(project_id: uuid.UUID, body: ProjectChatRequest, db: AsyncSession = Depends(get_db), user: User = Depends(get_current_user)):
    if _is_greeting(body.message):
        await _authorize_project_chat(db, project_id, user)
        return ProjectChatResponse(
            message={"role": "assistant", "content": "Hello! Ask me about themes, patterns, or evidence across this project's clusters."},
            model="deterministic",
            sources=[],
        )
    sources, messages, direct_answer = await _context(db, project_id, user, body)
    if direct_answer is not None:
        return ProjectChatResponse(message={"role": "assistant", "content": direct_answer}, model="tool", sources=sources)
    try:
        answer = await run_in_threadpool(
            chat_completion, base_url=settings.ollama_base_url, model=settings.ollama_chat_model,
            messages=messages, timeout=settings.ollama_timeout_seconds,
        )
    except OllamaChatError as exc:
        raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail=str(exc)) from exc
    return ProjectChatResponse(message={"role": "assistant", "content": answer}, model=settings.ollama_chat_model, sources=sources)


@router.post("/{project_id}/chat/stream")
async def project_chat_stream(project_id: uuid.UUID, body: ProjectChatRequest, db: AsyncSession = Depends(get_db), user: User = Depends(get_current_user)):
    if _is_greeting(body.message):
        await _authorize_project_chat(db, project_id, user)

        def greeting_events():
            yield json.dumps({"type": "token", "content": "Hello! Ask me about themes, patterns, or evidence across this project's clusters."}) + "\n"
            yield json.dumps({"type": "done", "model": "deterministic"}) + "\n"

        return StreamingResponse(greeting_events(), media_type="application/x-ndjson")
    sources, messages, direct_answer = await _context(db, project_id, user, body)

    def events():
        yield json.dumps({"type": "sources", "sources": [source.model_dump(mode="json") for source in sources]}) + "\n"
        if direct_answer is not None:
            yield json.dumps({"type": "token", "content": direct_answer}) + "\n"
            yield json.dumps({"type": "done", "model": "tool"}) + "\n"
            return
        try:
            for content in stream_chat_completion(
                base_url=settings.ollama_base_url, model=settings.ollama_chat_model,
                messages=messages, timeout=settings.ollama_timeout_seconds,
            ):
                yield json.dumps({"type": "token", "content": content}) + "\n"
            yield json.dumps({"type": "done", "model": settings.ollama_chat_model}) + "\n"
        except OllamaChatError as exc:
            yield json.dumps({"type": "error", "message": str(exc)}) + "\n"

    return StreamingResponse(events(), media_type="application/x-ndjson")
