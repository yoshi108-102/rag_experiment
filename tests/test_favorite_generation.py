from __future__ import annotations

from src.evals.favorite_generation import collect_favorite_cases, generate_cases_from_favorites
from src.evals.workbench import ensure_case_defaults


def _build_case(case_id: str, *, favorite: bool) -> dict:
    return ensure_case_defaults(
        {
            "case_id": case_id,
            "source": {"is_custom": False},
            "input": {"context": [{"role": "assistant", "content": "前提"}], "user_input": "棒の矯正が難しい"},
            "output": {"assistant_output": "どの瞬間が難しい？", "predicted_route": "CLARIFY"},
            "metadata": {"favorite": favorite, "dataset_type": "route_eval"},
            "labels": {},
        }
    )


def test_collect_favorite_cases_filters_by_flag():
    cases = [
        _build_case("c1", favorite=True),
        _build_case("c2", favorite=False),
        _build_case("c3", favorite=True),
    ]

    favorites = collect_favorite_cases(cases)

    assert [case["case_id"] for case in favorites] == ["c1", "c3"]


def test_generate_cases_from_favorites_creates_requested_drafts():
    favorites = [
        _build_case("fav-1", favorite=True),
        _build_case("fav-2", favorite=True),
    ]

    generated, summary = generate_cases_from_favorites(
        favorites,
        total_count=10,
        seed=7,
        batch_id="batch-test",
    )

    assert len(generated) == 10
    assert summary["generated"] == 10
    assert summary["favorite_count"] == 2
    assert summary["batch_id"] == "batch-test"
    assert set(summary["parent_case_ids"]) == {"fav-1", "fav-2"}

    for case in generated:
        metadata = case["metadata"]
        labels = case["labels"]
        source = case["source"]

        assert source["is_custom"] is True
        assert source["generated_from_favorite"] is True
        assert metadata["favorite"] is False
        assert metadata["edited"] is False
        assert metadata["generation"]["batch_id"] == "batch-test"
        assert labels["expected_route"] is None
        assert labels["label_status"] == "unlabeled"
        assert case["input"]["user_input"]
