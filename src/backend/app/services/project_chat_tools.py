from __future__ import annotations

from typing import Any

from app.project_chat_schemas import ProjectChatMessage
from app.services.ollama_chat import chat_message


PROJECT_CHAT_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "describe_project_chat",
            "description": "Describe what project chat can help with. Use for greetings, introductions, or questions about the assistant's capabilities.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "explain_project_data_source",
            "description": "Explain how the assistant knows a project-level fact, including cluster counts, sizes, sentiment, or ratings. Use for questions such as 'how do you know that?' or 'where does this number come from?'.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_project_cluster_count",
            "description": "Return the exact number of non-noise clusters in the current project. Use for questions asking how many clusters the project has.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "rank_clusters",
            "description": "Return an exact project-wide ranking of clusters by size, sentiment score, or rating. Use this for largest, smallest, highest, lowest, most positive, or most negative cluster questions.",
            "parameters": {
                "type": "object",
                "properties": {
                    "metric": {"type": "string", "enum": ["size", "sentiment", "rating"]},
                    "direction": {"type": "string", "enum": ["asc", "desc"], "description": "asc means the smallest or lowest numeric value; desc means the largest or highest numeric value."},
                    "limit": {"type": "integer", "minimum": 1, "maximum": 10},
                },
                "required": ["metric", "direction"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_related_clusters",
            "description": "List distinct clusters related to a topic or concept. Use only when the user asks which clusters relate to a subject. Do not use this to explain why one named cluster relates to a topic. Pass a concise topic query such as 'food' or 'restaurants', not the full user question.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "limit": {"type": "integer", "minimum": 1, "maximum": 15},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "explain_cluster_relation",
            "description": "Explain why one named cluster is related to a topic, using evidence only from that cluster. Use when the user says why/how/what evidence or refers to a specific cluster. Never return a list of additional clusters.",
            "parameters": {
                "type": "object",
                "properties": {
                    "cluster_reference": {"type": "string", "description": "The exact cluster label or UUID named by the user."},
                    "topic": {"type": "string", "description": "The concise topic to explain the relation to."},
                    "limit": {"type": "integer", "minimum": 1, "maximum": 10},
                },
                "required": ["cluster_reference", "topic"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_project_evidence",
            "description": "Semantically search clustered project evidence for qualitative questions about themes, experiences, recurring issues, contradictions, or examples.",
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"],
            },
        },
    },
]


def plan_project_tool(*, base_url: str, model: str, timeout: int, question: str, history: list[ProjectChatMessage]) -> tuple[str, dict[str, Any]]:
    messages = [{
        "role": "system",
        "content": """Choose exactly one tool for the user's request. Never answer from memory.
Use rank_clusters for any project-wide numerical cluster ranking. Use search_related_clusters only when asked to list clusters related to a topic. Use explain_cluster_relation when asked why/how a named cluster relates to a topic. Use search_project_evidence for qualitative review evidence questions.

Examples:
- "largest cluster" -> rank_clusters(metric="size", direction="desc", limit=1)
- "smallest cluster" -> rank_clusters(metric="size", direction="asc", limit=1)
- "lowest sentiment" -> rank_clusters(metric="sentiment", direction="asc", limit=1)
- "highest rating" -> rank_clusters(metric="rating", direction="desc", limit=1)
- "how many clusters does this project have?" -> get_project_cluster_count()
- "why is beer / drinks / wine related to sport?" -> explain_cluster_relation(cluster_reference="beer / drinks / wine", topic="sport", limit=5)""",
    }]
    messages.extend({"role": item.role, "content": item.content} for item in history[-8:])
    messages.append({"role": "user", "content": question})
    response = chat_message(base_url=base_url, model=model, messages=messages, timeout=timeout, tools=PROJECT_CHAT_TOOLS)
    name, arguments = _tool_call_from_response(response)
    proposed = _validate_tool_call(name, arguments)
    return validate_project_tool(
        base_url=base_url, model=model, timeout=timeout, question=question, proposed=proposed,
    )


def validate_project_tool(*, base_url: str, model: str, timeout: int, question: str, proposed: tuple[str, dict[str, Any]]) -> tuple[str, dict[str, Any]]:
    name, arguments = proposed
    response = chat_message(
        base_url=base_url,
        model=model,
        timeout=timeout,
        tools=PROJECT_CHAT_TOOLS,
        messages=[
            {
                "role": "system",
                "content": """Validate the proposed project-chat tool call against the user question. Return exactly one corrected tool call.
For rank_clusters, asc means smallest/lowest and desc means largest/highest. Use get_project_cluster_count only for "how many clusters". Use explain_project_data_source for how/why the assistant knows a project fact. Use describe_project_chat for greetings/capabilities. Use explain_cluster_relation, not search_related_clusters, when the user asks why/how a named cluster relates to a topic. Do not answer the question yourself.
Examples: largest cluster = size/desc; smallest cluster = size/asc; lowest sentiment = sentiment/asc; highest rating = rating/desc; how many clusters = get_project_cluster_count; how do you know the count = explain_project_data_source; hello what can you do = describe_project_chat; why beer / drinks / wine relates to sport = explain_cluster_relation.""",
            },
            {"role": "user", "content": f"Question: {question}\nProposed tool: {name}({arguments})"},
        ],
    )
    validated_name, validated_arguments = _tool_call_from_response(response)
    return _validate_tool_call(validated_name, validated_arguments)


def _tool_call_from_response(response: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    calls = response.get("tool_calls")
    if not isinstance(calls, list) or not calls:
        raise ValueError("The chat planner did not select a tool.")
    function = calls[0].get("function") if isinstance(calls[0], dict) else None
    if not isinstance(function, dict):
        raise ValueError("The chat planner returned an invalid tool call.")
    name, arguments = function.get("name"), function.get("arguments", {})
    if not isinstance(name, str) or not isinstance(arguments, dict):
        raise ValueError("The chat planner returned invalid tool arguments.")
    return name, arguments


def _validate_tool_call(name: str, arguments: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    if name in {"describe_project_chat", "explain_project_data_source"}:
        return name, {}
    if name == "get_project_cluster_count":
        return name, {}
    if name == "rank_clusters":
        metric = arguments.get("metric")
        direction = arguments.get("direction")
        if metric not in {"size", "sentiment", "rating"} or direction not in {"asc", "desc"}:
            raise ValueError("The chat planner requested an unsupported cluster ranking.")
        try:
            limit = int(float(arguments.get("limit", 1)))
        except (TypeError, ValueError, OverflowError):
            # `limit` is optional; a model-generated spelling or null must not
            # reject an otherwise valid exact ranking request.
            limit = 1
        return name, {"metric": metric, "direction": direction, "limit": max(1, min(limit, 10))}
    if name == "search_project_evidence":
        return name, _validate_query_arguments(arguments, max_limit=10)
    if name == "search_related_clusters":
        return name, _validate_query_arguments(arguments, max_limit=15)
    if name == "explain_cluster_relation":
        cluster_reference = arguments.get("cluster_reference")
        topic = arguments.get("topic")
        if not isinstance(cluster_reference, str) or not cluster_reference.strip() or not isinstance(topic, str) or not topic.strip():
            raise ValueError("The chat planner requested an invalid cluster explanation.")
        query_arguments = _validate_query_arguments({"query": topic, "limit": arguments.get("limit", 5)}, max_limit=10)
        return name, {"cluster_reference": cluster_reference.strip()[:500], "topic": query_arguments["query"], "limit": query_arguments["limit"]}
    raise ValueError("The chat planner requested an unsupported tool.")


def _validate_query_arguments(arguments: dict[str, Any], max_limit: int) -> dict[str, Any]:
    query = arguments.get("query")
    if not isinstance(query, str) or not query.strip():
        raise ValueError("The chat planner requested an empty evidence search.")
    try:
        limit = int(float(arguments.get("limit", max_limit)))
    except (TypeError, ValueError, OverflowError):
        limit = max_limit
    return {"query": query.strip()[:4000], "limit": max(1, min(limit, max_limit))}
