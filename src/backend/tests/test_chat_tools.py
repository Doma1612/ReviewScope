import json
import uuid
from types import SimpleNamespace

from app.api.projects import (
    _chat_result_from_doc,
    _chat_review_text_by_document,
    _looks_like_tool_call_content,
    _social_chat_answer,
    _tool_filter_matches,
    _tool_plan_from_message_content,
    _validate_cluster_chat_plan,
)


def test_bottom_review_tool_keeps_rating_ascending():
    plan = _validate_cluster_chat_plan(
        {
            "tool": "list_reviews",
            "args": {"limit": 5, "sort_by": "rating", "sort_dir": "asc"},
            "response_mode": "direct_list",
        },
    )

    assert plan == {
        "tool": "list_reviews",
        "args": {"limit": 5, "sort_by": "rating", "sort_dir": "asc", "filters": []},
        "response_mode": "direct_list",
    }


def test_count_plan_preserves_rating_filter_for_tool_memory():
    plan = _validate_cluster_chat_plan(
        {
            "tool": "count_reviews",
            "args": {"filters": [{"field": "stars", "op": "eq", "value": 1}]},
            "response_mode": "count",
        },
    )

    assert plan == {
        "tool": "count_reviews",
        "args": {"filters": [{"field": "rating", "op": "eq", "value": 1}]},
        "response_mode": "count",
    }
    assert _tool_filter_matches(1.0, "eq", 1)
    assert not _tool_filter_matches(5.0, "eq", 1)


def test_planner_cannot_override_cluster_scope():
    assert _validate_cluster_chat_plan(
        {
            "tool": "list_reviews",
            "args": {"cluster_id": "another-cluster", "limit": 5},
        },
    ) == {
        "tool": "list_reviews",
        "args": {"limit": 5, "filters": [], "sort_by": "rating", "sort_dir": "desc"},
        "response_mode": "direct_list",
    }


def test_cluster_analysis_is_a_valid_tool_choice():
    assert _validate_cluster_chat_plan(
        {"tool": "analyze_current_cluster", "args": {"focus": "sentiment"}, "response_mode": "analysis"},
    ) == {"tool": "analyze_current_cluster", "args": {"focus": "sentiment"}, "response_mode": "analysis"}


def test_single_filter_object_is_accepted():
    assert _validate_cluster_chat_plan(
        {
            "tool": "count_reviews",
            "args": {"filters": {"field": "rating", "op": "eq", "value": 1}},
            "response_mode": "count",
        },
    )["args"]["filters"] == [{"field": "rating", "op": "eq", "value": 1}]


def test_exact_rating_tool_normalizes_to_list_filter():
    assert _validate_cluster_chat_plan(
        {
            "tool": "list_reviews_by_rating",
            "args": {"stars": 1, "limit": 5},
            "response_mode": "direct_list",
        }
    ) == {
        "tool": "list_reviews",
        "args": {
            "limit": 5,
            "sort_by": "rating",
            "sort_dir": "desc",
            "filters": [{"field": "rating", "op": "eq", "value": 1.0}],
        },
        "response_mode": "direct_list",
    }


def test_lowest_rating_tool_normalizes_to_rating_ascending():
    assert _validate_cluster_chat_plan(
        {"tool": "list_lowest_rated_reviews", "args": {"limit": 5}, "response_mode": "direct_list"}
    )["args"] == {"limit": 5, "sort_by": "rating", "sort_dir": "asc", "filters": []}


def test_field_value_at_top_level_becomes_filter():
    assert _validate_cluster_chat_plan(
        {
            "tool": "list_reviews",
            "args": {"field": "stars", "value": 5, "limit": 5},
            "response_mode": "direct_list",
        }
    )["args"]["filters"] == [{"field": "rating", "op": "eq", "value": 5}]


def test_content_tool_call_is_recovered_from_ollama_json_text():
    assert _tool_plan_from_message_content(
        '{"name":"count_reviews","parameters":{"filters":[{"field":"rating","op":"eq","value":null}]}}'
    ) == {
        "tool": "count_reviews",
        "args": {"filters": [{"field": "rating", "op": "eq", "value": None}]},
        "response_mode": "direct_list",
    }


def test_null_filter_is_treated_as_no_filter_for_cluster_count():
    assert _validate_cluster_chat_plan(
        {
            "tool": "count_reviews",
            "args": {"filters": [{"field": "rating", "op": "eq", "value": None}]},
        }
    ) == {
        "tool": "count_reviews",
        "args": {"filters": []},
        "response_mode": "direct_list",
    }


def test_empty_filter_object_is_treated_as_no_filter_for_cluster_count():
    assert _validate_cluster_chat_plan(
        {"tool": "count_reviews", "args": {"filters": {}}}
    )["args"]["filters"] == []


def test_total_count_tool_discards_unexpected_or_stale_filters():
    assert _validate_cluster_chat_plan(
        {
            "tool": "count_current_cluster_reviews",
            "args": {"filters": [{"field": "rating", "op": "eq", "value": 1}]},
        }
    ) == {
        "tool": "count_reviews",
        "args": {"filters": []},
        "response_mode": "direct_list",
    }


def test_cluster_size_tool_alias_normalizes_to_count():
    assert _validate_cluster_chat_plan({"tool": "get_cluster_size", "args": {}})["tool"] == "count_reviews"


def test_chat_result_uses_complete_review_text_shown_in_table():
    doc = SimpleNamespace(id=uuid.uuid4(), primary_key_value="review-1", text="Complete review text")
    review_text = _chat_review_text_by_document([doc])

    result = _chat_result_from_doc(doc, 5, review_text[doc.id])

    assert result.snippet == "Complete review text"


def test_social_messages_are_answered_without_cluster_analysis():
    assert _social_chat_answer("hello") == "Hello! What would you like to know about this cluster?"
    assert _social_chat_answer("Hi!") == "Hello! What would you like to know about this cluster?"
    assert _social_chat_answer("thank you") == "You're welcome!"
    assert _social_chat_answer("How many reviews are there?") is None


def test_double_encoded_tool_call_is_recovered():
    content = json.dumps(json.dumps({"name": "count_reviews", "parameters": {"filters": {}}}))

    assert _tool_plan_from_message_content(content) == {
        "tool": "count_reviews",
        "args": {"filters": {}},
        "response_mode": "direct_list",
    }


def test_malformed_tool_protocol_is_not_safe_to_display():
    assert _looks_like_tool_call_content('{"name":"count_reviews","parameters":')
    assert not _looks_like_tool_call_content("There are 10 reviews in this cluster.")
