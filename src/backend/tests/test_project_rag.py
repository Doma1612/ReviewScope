from app.project_chat_schemas import ProjectChatMessage, ProjectChatSource
from app.services import project_chat_tools
from app.services.project_chat_tools import _validate_tool_call
from app.services.project_rag import rag_messages


def test_rag_messages_limit_history_and_label_evidence() -> None:
    source = ProjectChatSource(
        citation_id="S1",
        unit="segment",
        document_id="00000000-0000-0000-0000-000000000001",
        segment_id="00000000-0000-0000-0000-000000000002",
        primary_key_value="review-1",
        cluster_id="00000000-0000-0000-0000-000000000003",
        cluster_label="Cleanliness",
        snippet="The bathroom floor was dirty on arrival.",
        distance=0.12,
    )
    history = [ProjectChatMessage(role="user", content=str(index)) for index in range(10)]

    messages = rag_messages("What cleanliness problems recur?", history, [source])

    assert "[S1]" in messages[0]["content"]
    assert "Cluster: Cleanliness" in messages[0]["content"]
    assert "Evidence: The bathroom floor was dirty on arrival." in messages[0]["content"]
    assert len(messages) == 10
    assert messages[1]["content"] == "2"
    assert messages[-1] == {"role": "user", "content": "What cleanliness problems recur?"}


def test_rank_tool_arguments_are_allow_listed_and_clamped() -> None:
    assert _validate_tool_call("describe_project_chat", {"ignored": "value"}) == ("describe_project_chat", {})
    assert _validate_tool_call("explain_project_data_source", {"ignored": "value"}) == ("explain_project_data_source", {})
    assert _validate_tool_call("get_project_cluster_count", {"ignored": "value"}) == ("get_project_cluster_count", {})
    assert _validate_tool_call("rank_clusters", {"metric": "size", "direction": "desc", "limit": 50}) == (
        "rank_clusters", {"metric": "size", "direction": "desc", "limit": 10}
    )
    assert _validate_tool_call("rank_clusters", {"metric": "sentiment", "direction": "asc", "limit": "one"}) == (
        "rank_clusters", {"metric": "sentiment", "direction": "asc", "limit": 1}
    )
    assert _validate_tool_call("search_related_clusters", {"query": "food", "limit": 30}) == (
        "search_related_clusters", {"query": "food", "limit": 15}
    )
    assert _validate_tool_call("explain_cluster_relation", {"cluster_reference": "beer / drinks / wine", "topic": "sport", "limit": 5}) == (
        "explain_cluster_relation", {"cluster_reference": "beer / drinks / wine", "topic": "sport", "limit": 5}
    )


def test_cluster_fact_prompt_preserves_exact_values() -> None:
    source = ProjectChatSource(
        citation_id="C1",
        unit="cluster",
        cluster_id="00000000-0000-0000-0000-000000000003",
        cluster_label="minutes / wait / table",
        size=494,
        n_mentions=810,
        sentiment_avg=-0.9,
        mean_stars=2.1,
        snippet="minutes / wait / table: 494 reviews; sentiment -0.90",
    )

    messages = rag_messages("What is the largest cluster?", [], [source])

    assert "Size: 494 reviews" in messages[0]["content"]
    assert "Sentiment: -0.9" in messages[0]["content"]
    assert "[C1]" in messages[0]["content"]


def test_validator_corrects_a_largest_cluster_direction(monkeypatch) -> None:
    monkeypatch.setattr(
        project_chat_tools,
        "chat_message",
        lambda **_kwargs: {"tool_calls": [{"function": {"name": "rank_clusters", "arguments": {"metric": "size", "direction": "desc", "limit": 1}}}]},
    )

    result = project_chat_tools.validate_project_tool(
        base_url="http://ollama", model="test", timeout=1,
        question="What is the largest cluster?",
        proposed=("rank_clusters", {"metric": "size", "direction": "asc", "limit": 1}),
    )

    assert result == ("rank_clusters", {"metric": "size", "direction": "desc", "limit": 1})
