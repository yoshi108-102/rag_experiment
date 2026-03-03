"""chat_sessions JSONL から eval 用データセットJSONLを作成するCLI。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.evals.log_to_eval import (
    build_eval_dataset,
    list_jsonl_files,
    parse_route_quota,
    write_jsonl,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="chat logs -> eval dataset builder")
    parser.add_argument(
        "--logs",
        default="logs/chat_sessions",
        help="入力chat session JSONLのディレクトリまたはファイル",
    )
    parser.add_argument(
        "--out",
        default="evals/cases/route_eval_candidates.jsonl",
        help="出力先JSONLファイル",
    )
    parser.add_argument("--max-cases", type=int, default=100, help="最大ケース数")
    parser.add_argument(
        "--context-turns",
        type=int,
        default=2,
        help="user入力前に保持する過去ターン数",
    )
    parser.add_argument(
        "--min-user-chars",
        type=int,
        default=4,
        help="ユーザー入力の最小文字数（正規化後）",
    )
    parser.add_argument(
        "--dedupe-mode",
        choices=["user_only", "user_and_route", "user_and_response"],
        default="user_and_route",
        help="重複除去キー",
    )
    parser.add_argument("--seed", type=int, default=42, help="サンプリングseed")
    parser.add_argument(
        "--route-quota",
        default=None,
        help="route配分。例: CLARIFY=50,DEEPEN=20,FINISH=20,PARK=10",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    log_paths = list_jsonl_files(Path(args.logs))
    if not log_paths:
        raise SystemExit(f"No JSONL files found: {args.logs}")

    route_quota = parse_route_quota(args.route_quota)
    cases, result = build_eval_dataset(
        log_paths,
        max_cases=args.max_cases,
        context_turns=args.context_turns,
        min_user_chars=args.min_user_chars,
        dedupe_mode=args.dedupe_mode,
        seed=args.seed,
        route_quota=route_quota,
    )

    out_path = Path(args.out)
    write_jsonl(cases, out_path)
    summary = {
        "input_log_files": len(log_paths),
        "total_candidates": result.total_candidates,
        "after_dedup": result.after_dedup,
        "selected": result.selected,
        "route_counts": result.route_counts,
        "output_path": str(out_path),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
