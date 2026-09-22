import os

os.environ.setdefault("SECRET_KEY", "test-secret")

from app.api.project_chat import _is_greeting, _ranking_answer, _related_clusters_answer
from app.project_chat_schemas import ProjectChatSource


def test_project_chat_greetings_bypass_retrieval() -> None:
    assert _is_greeting("hello")
    assert _is_greeting("Hi there!")
    assert _is_greeting(" hey ")
    assert not _is_greeting("hello, what are the largest clusters?")
    assert not _is_greeting("what clusters discuss breakfast?")


def test_exact_cluster_tools_do_not_depend_on_a_second_llm_answer() -> None:
    source = ProjectChatSource(
        citation_id="C1",
        unit="cluster",
        cluster_id="00000000-0000-0000-0000-000000000003",
        cluster_label="gym / class / fitness",
        size=36,
        snippet="gym / class / fitness: 36 reviews",
    )

    assert _ranking_answer([source], "size", "asc") == "The smallest cluster is **gym / class / fitness**: 36 reviews [C1]."
    assert _related_clusters_answer([source]) == "Related clusters:\n- **gym / class / fitness** [C1]"
