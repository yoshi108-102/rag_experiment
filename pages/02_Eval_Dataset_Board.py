from __future__ import annotations

from collections import Counter
import hashlib
from pathlib import Path
from typing import Any

import streamlit as st

from src.evals.log_to_eval import extract_eval_case_drafts, list_jsonl_files
from src.evals.workbench import (
    DATASET_TYPE_OPTIONS,
    ROUTE_OPTIONS,
    build_custom_case,
    delete_case_from_state,
    ensure_case_defaults,
    export_cases_to_jsonl,
    load_workbench_state,
    merge_base_cases_with_state,
    parse_context_lines,
    render_context_lines,
    save_workbench_state,
    upsert_case_in_state,
)


DEFAULT_LOGS_PATH = "logs/chat_sessions"
DEFAULT_STATE_PATH = "evals/workbench/state.json"
DEFAULT_EXPORT_PATH = "evals/cases/route_eval_edited.jsonl"


def main() -> None:
    st.title("Eval Dataset Board")
    st.caption("会話ログからケースを取得し、編集・ラベル付け・エクスポートするワークベンチ")

    with st.sidebar:
        st.subheader("Data Source")
        logs_path_str = st.text_input("ログディレクトリ", value=DEFAULT_LOGS_PATH)
        state_path_str = st.text_input("編集状態ファイル", value=DEFAULT_STATE_PATH)
        min_user_chars = int(st.number_input("最小ユーザー文字数", min_value=1, max_value=200, value=4))
        context_turns = int(st.number_input("保持する過去ターン数", min_value=0, max_value=10, value=2))

    logs_path = Path(logs_path_str)
    state_path = Path(state_path_str)
    state = load_workbench_state(state_path)

    log_files = list_jsonl_files(logs_path)
    file_options = [str(path) for path in log_files]
    selected_file_options = st.sidebar.multiselect(
        "履歴一覧（読み込むログ）",
        options=file_options,
        default=file_options,
    )
    selected_logs = [Path(path) for path in selected_file_options]

    base_cases = _load_base_cases(
        selected_logs,
        context_turns=context_turns,
        min_user_chars=min_user_chars,
    )
    merged_cases = merge_base_cases_with_state(base_cases, state)
    merged_cases = [ensure_case_defaults(case) for case in merged_cases]

    with st.sidebar:
        st.subheader("Board Filter")
        route_filter = st.multiselect(
            "route",
            options=[route for route in ROUTE_OPTIONS if route],
            default=[route for route in ROUTE_OPTIONS if route],
        )
        edited_filter = st.selectbox(
            "編集状態",
            options=["all", "edited_only", "unedited_only"],
            index=0,
        )
        search_query = st.text_input("キーワード検索", value="")
        board_columns = int(st.slider("ボード列数", min_value=1, max_value=3, value=2))
        board_limit = int(st.slider("表示件数", min_value=1, max_value=300, value=60))

    filtered_cases = _filter_cases(
        merged_cases,
        routes=route_filter,
        edited_filter=edited_filter,
        query=search_query,
    )[:board_limit]

    _render_summary(merged_cases, filtered_cases, len(selected_logs))
    _render_custom_case_creator(state, state_path)
    _render_export_controls(merged_cases, filtered_cases)

    st.divider()
    st.subheader("Case Board")
    if not filtered_cases:
        st.info("条件に一致するケースがありません。")
        return

    for row_cases in _chunk(filtered_cases, board_columns):
        columns = st.columns(board_columns)
        for col, case in zip(columns, row_cases, strict=False):
            with col:
                _render_case_card(case, state, state_path)


def _load_base_cases(
    log_paths: list[Path],
    *,
    context_turns: int,
    min_user_chars: int,
) -> list[dict[str, Any]]:
    drafts = extract_eval_case_drafts(
        log_paths,
        context_turns=context_turns,
        min_user_chars=min_user_chars,
    )
    cases = [draft.to_eval_case() for draft in drafts]
    cases.sort(
        key=lambda case: str((case.get("source") or {}).get("assistant_timestamp") or ""),
        reverse=True,
    )
    return cases


def _render_summary(all_cases: list[dict[str, Any]], visible_cases: list[dict[str, Any]], selected_logs: int) -> None:
    route_counts = Counter(str((case.get("output") or {}).get("predicted_route") or "") for case in all_cases)
    edited_count = sum(
        1 for case in all_cases if bool((case.get("metadata") or {}).get("edited"))
    )
    st.write(
        {
            "selected_logs": selected_logs,
            "total_cases": len(all_cases),
            "visible_cases": len(visible_cases),
            "edited_cases": edited_count,
            "route_counts": {k: v for k, v in route_counts.items() if k},
        }
    )


def _render_custom_case_creator(state: dict[str, Any], state_path: Path) -> None:
    with st.expander("新規テストケースを作成", expanded=False):
        with st.form("create_custom_case_form", clear_on_submit=True):
            st.caption("任意の会話を自作し、テストデータとして追加できます。")
            dataset_type = st.selectbox("データセット種別", options=DATASET_TYPE_OPTIONS, index=0)
            context_raw = st.text_area(
                "コンテキスト（任意）",
                value="",
                help="1行ごとに `user: ...` / `assistant: ...` の形式で入力",
                height=100,
            )
            user_input = st.text_area("ユーザー入力", value="", height=120)
            assistant_output = st.text_area("アシスタント出力", value="", height=120)
            predicted_route = st.selectbox(
                "predicted_route（任意）",
                options=ROUTE_OPTIONS,
                index=0,
            )
            expected_route = st.selectbox(
                "expected_route（任意）",
                options=ROUTE_OPTIONS,
                index=0,
            )
            label_note = st.text_input("メモ（任意）", value="")
            edited = st.checkbox("編集済みとして追加", value=True)
            submitted = st.form_submit_button("ケースを追加")

        if submitted:
            if not user_input.strip():
                st.error("ユーザー入力は必須です。")
                return
            if not assistant_output.strip():
                st.error("アシスタント出力は必須です。")
                return

            case = build_custom_case(
                dataset_type=dataset_type,
                user_input=user_input.strip(),
                assistant_output=assistant_output.strip(),
                context=parse_context_lines(context_raw),
                predicted_route=predicted_route or None,
                expected_route=expected_route or None,
                label_note=label_note or None,
            )
            case["metadata"]["edited"] = edited
            upsert_case_in_state(case, state)
            save_workbench_state(state_path, state)
            st.success(f"追加しました: {case['case_id']}")
            st.rerun()


def _render_export_controls(
    all_cases: list[dict[str, Any]],
    visible_cases: list[dict[str, Any]],
) -> None:
    with st.expander("JSONL エクスポート", expanded=False):
        out_path_str = st.text_input("出力先", value=DEFAULT_EXPORT_PATH)
        scope = st.selectbox("出力対象", options=["visible_only", "all_cases"], index=0)
        only_edited = st.checkbox("編集済みのみを出力", value=True)

        dataset_types = sorted(
            {
                str((case.get("metadata") or {}).get("dataset_type") or "")
                for case in all_cases
                if str((case.get("metadata") or {}).get("dataset_type") or "")
            }
        )
        selected_types = st.multiselect(
            "データセット種別フィルタ（空なら全件）",
            options=dataset_types,
            default=dataset_types,
        )

        if st.button("エクスポート実行", type="primary"):
            target_cases = visible_cases if scope == "visible_only" else all_cases
            exported = export_cases_to_jsonl(
                target_cases,
                Path(out_path_str),
                only_edited=only_edited,
                dataset_types=selected_types,
            )
            st.success(f"{exported} 件を書き出しました: {out_path_str}")


def _render_case_card(case: dict[str, Any], state: dict[str, Any], state_path: Path) -> None:
    case = ensure_case_defaults(case)
    case_id = str(case.get("case_id"))
    widget_suffix = hashlib.sha1(case_id.encode("utf-8")).hexdigest()[:10]

    source = case.get("source") or {}
    input_block = case.get("input") or {}
    output_block = case.get("output") or {}
    metadata = case.get("metadata") or {}
    labels = case.get("labels") or {}
    is_custom = bool(source.get("is_custom"))

    with st.container(border=True):
        with st.form(f"case_form_{widget_suffix}", clear_on_submit=False):
            left, right = st.columns([0.78, 0.22])
            with left:
                st.markdown(f"**{case_id}**")
                st.caption(f"route: `{output_block.get('predicted_route', '') or '-'}`")
                if source.get("assistant_timestamp"):
                    st.caption(f"time: {source['assistant_timestamp']}")
            with right:
                edited = st.checkbox(
                    "編集済み",
                    value=bool(metadata.get("edited")),
                    key=f"edited_{widget_suffix}",
                )

            st.caption(f"log: {source.get('log_path') or '(custom)'}")

            current_dataset_type = str(metadata.get("dataset_type") or "route_eval")
            preset_index = (
                DATASET_TYPE_OPTIONS.index(current_dataset_type)
                if current_dataset_type in DATASET_TYPE_OPTIONS
                else DATASET_TYPE_OPTIONS.index("custom")
            )
            selected_dataset_type = st.selectbox(
                "データセット種別",
                options=DATASET_TYPE_OPTIONS,
                index=preset_index,
                key=f"dtype_{widget_suffix}",
            )
            custom_dataset_type = st.text_input(
                "カスタム種別（任意）",
                value="" if current_dataset_type in DATASET_TYPE_OPTIONS else current_dataset_type,
                key=f"dtype_custom_{widget_suffix}",
            )
            final_dataset_type = custom_dataset_type.strip() or selected_dataset_type

            context_text = st.text_area(
                "Context（編集可）",
                value=render_context_lines(input_block.get("context") or []),
                height=90,
                key=f"context_{widget_suffix}",
                help="1行ごとに `user:` / `assistant:` を指定",
            )
            user_input = st.text_area(
                "User Input",
                value=str(input_block.get("user_input") or ""),
                height=110,
                key=f"user_input_{widget_suffix}",
            )
            assistant_output = st.text_area(
                "Assistant Output",
                value=str(output_block.get("assistant_output") or ""),
                height=110,
                key=f"assistant_output_{widget_suffix}",
            )

            predicted_route = st.selectbox(
                "predicted_route",
                options=ROUTE_OPTIONS,
                index=_route_index(str(output_block.get("predicted_route") or "")),
                key=f"pred_route_{widget_suffix}",
            )
            expected_route = st.selectbox(
                "expected_route",
                options=ROUTE_OPTIONS,
                index=_route_index(str(labels.get("expected_route") or "")),
                key=f"exp_route_{widget_suffix}",
            )
            label_note = st.text_input(
                "label_note",
                value=str(labels.get("label_note") or ""),
                key=f"label_note_{widget_suffix}",
            )
            labeler = st.text_input(
                "labeler",
                value=str(labels.get("labeler") or ""),
                key=f"labeler_{widget_suffix}",
            )

            save_col, delete_col = st.columns([0.6, 0.4])
            with save_col:
                save_clicked = st.form_submit_button("保存", type="primary")
            with delete_col:
                delete_clicked = st.form_submit_button(
                    "カスタム削除",
                    disabled=not is_custom,
                )

        if delete_clicked and is_custom:
            delete_case_from_state(case_id, state)
            save_workbench_state(state_path, state)
            st.success(f"削除しました: {case_id}")
            st.rerun()

        if save_clicked:
            if not user_input.strip():
                st.error("User Input は空にできません。")
                return
            if not assistant_output.strip():
                st.error("Assistant Output は空にできません。")
                return

            output_block["assistant_output"] = assistant_output.strip()
            output_block["predicted_route"] = predicted_route or ""
            input_block["user_input"] = user_input.strip()
            input_block["context"] = parse_context_lines(context_text)

            metadata["dataset_type"] = final_dataset_type
            metadata["edited"] = bool(edited)

            labels["expected_route"] = expected_route or None
            labels["label_status"] = "labeled" if expected_route else "unlabeled"
            labels["label_note"] = label_note.strip() or None
            labels["labeler"] = labeler.strip() or None

            case["output"] = output_block
            case["input"] = input_block
            case["metadata"] = metadata
            case["labels"] = labels

            upsert_case_in_state(case, state)
            save_workbench_state(state_path, state)
            st.success(f"保存しました: {case_id}")
            st.rerun()


def _filter_cases(
    cases: list[dict[str, Any]],
    *,
    routes: list[str],
    edited_filter: str,
    query: str,
) -> list[dict[str, Any]]:
    normalized_query = query.strip().lower()
    selected_routes = set(routes)

    filtered: list[dict[str, Any]] = []
    for case in cases:
        output_block = case.get("output") or {}
        route = str(output_block.get("predicted_route") or "")
        if selected_routes and route not in selected_routes:
            continue

        metadata = case.get("metadata") or {}
        edited = bool(metadata.get("edited"))
        if edited_filter == "edited_only" and not edited:
            continue
        if edited_filter == "unedited_only" and edited:
            continue

        if normalized_query:
            blob = _build_search_blob(case)
            if normalized_query not in blob:
                continue
        filtered.append(case)
    return filtered


def _build_search_blob(case: dict[str, Any]) -> str:
    input_block = case.get("input") or {}
    output_block = case.get("output") or {}
    context = input_block.get("context") or []
    chunks = [
        str(input_block.get("user_input") or ""),
        str(output_block.get("assistant_output") or ""),
        str(output_block.get("predicted_route") or ""),
        str((case.get("metadata") or {}).get("dataset_type") or ""),
        str((case.get("labels") or {}).get("expected_route") or ""),
    ]
    for item in context:
        if not isinstance(item, dict):
            continue
        chunks.append(str(item.get("content") or ""))
    return "\n".join(chunks).lower()


def _route_index(route_value: str) -> int:
    return ROUTE_OPTIONS.index(route_value) if route_value in ROUTE_OPTIONS else 0


def _chunk(items: list[dict[str, Any]], size: int) -> list[list[dict[str, Any]]]:
    if size <= 0:
        return [items]
    return [items[idx : idx + size] for idx in range(0, len(items), size)]


if __name__ == "__main__":
    main()

