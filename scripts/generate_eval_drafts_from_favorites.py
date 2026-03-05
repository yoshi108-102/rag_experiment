"""favoriteケースから人手修正向けの下書きケースを生成するCLI。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.evals.favorite_generation import (
    DEFAULT_DRAFT_COUNT,
    FAVORITE_DEFAULT_EXPORT_DIR,
    GENERATED_DEFAULT_EXPORT_DIR,
    build_generation_batch_id,
    collect_favorite_cases,
    generate_cases_from_favorites,
    write_jsonl_cases,
)
from src.evals.workbench import load_workbench_state, save_workbench_state, upsert_case_in_state


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="favoriteケースから下書きデータを生成")
    parser.add_argument(
        "--state-path",
        default="evals/workbench/state.json",
        help="入力ワークベンチ状態JSON",
    )
    parser.add_argument(
        "--count",
        type=int,
        default=DEFAULT_DRAFT_COUNT,
        help="生成件数",
    )
    parser.add_argument("--seed", type=int, default=42, help="生成seed")
    parser.add_argument(
        "--favorite-out-dir",
        default=FAVORITE_DEFAULT_EXPORT_DIR,
        help="favoriteスナップショットJSONLの出力ディレクトリ",
    )
    parser.add_argument(
        "--generated-out-dir",
        default=GENERATED_DEFAULT_EXPORT_DIR,
        help="生成下書きJSONLの出力ディレクトリ",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    state_path = Path(args.state_path)
    state = load_workbench_state(state_path)

    raw_cases = state.get("cases") or {}
    if not isinstance(raw_cases, dict):
        raw_cases = {}

    favorite_cases = collect_favorite_cases(
        [case for case in raw_cases.values() if isinstance(case, dict)]
    )
    if not favorite_cases:
        raise SystemExit(f"favorite case not found in state: {state_path}")

    batch_id = build_generation_batch_id()
    generated_cases, generation_summary = generate_cases_from_favorites(
        favorite_cases,
        total_count=args.count,
        seed=args.seed,
        batch_id=batch_id,
    )

    for case in generated_cases:
        upsert_case_in_state(case, state)
    save_workbench_state(state_path, state)

    favorite_out_path = Path(args.favorite_out_dir) / f"favorite_snapshot_{batch_id}.jsonl"
    generated_out_path = Path(args.generated_out_dir) / f"generated_drafts_{batch_id}.jsonl"
    write_jsonl_cases(favorite_cases, favorite_out_path)
    write_jsonl_cases(generated_cases, generated_out_path)

    summary = {
        "state_path": str(state_path),
        "favorite_snapshot_path": str(favorite_out_path),
        "generated_path": str(generated_out_path),
        **generation_summary,
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
