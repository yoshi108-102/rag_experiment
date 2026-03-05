from __future__ import annotations

from collections import Counter
import hashlib
from pathlib import Path
from typing import Any

import streamlit as st

from src.evals.log_to_eval import extract_eval_case_drafts, list_jsonl_files
from src.evals.workbench import (
    CHAT_ROLE_OPTIONS,
    DATASET_TYPE_OPTIONS,
    ROUTE_OPTIONS,
    apply_conversation_to_case,
    build_custom_case,
    case_to_conversation,
    delete_conversation_messages_by_index,
    ensure_case_defaults,
    export_cases_to_jsonl,
    initial_user_question,
    load_workbench_state,
    merge_base_cases_with_state,
    normalize_conversation,
    parse_context_lines,
    save_workbench_state,
    upsert_case_in_state,
)


DEFAULT_LOGS_PATH = "logs/chat_sessions"
DEFAULT_STATE_PATH = "evals/workbench/state.json"
DEFAULT_EXPORT_PATH = "evals/cases/route_eval_edited.jsonl"
SELECTED_CASE_KEY = "eval_board_selected_case_id"


def main() -> None:
    st.title("Eval Dataset Board")
    st.caption("ダッシュボードでケースを選び、会話UIで編集して評価データを作成します。")

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
    case_map = {str(case.get("case_id")): case for case in merged_cases if case.get("case_id")}

    selected_case_id = st.session_state.get(SELECTED_CASE_KEY)
    if selected_case_id and selected_case_id not in case_map:
        st.session_state[SELECTED_CASE_KEY] = None
        selected_case_id = None

    if selected_case_id:
        case = case_map[selected_case_id]
        _render_conversation_editor(case, state, state_path)
        return

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
        favorite_filter = st.selectbox(
            "favorite",
            options=["all", "favorite_only", "non_favorite_only"],
            index=0,
        )
        dataset_type_filter = st.multiselect(
            "データセット種別",
            options=sorted(
                {
                    str((case.get("metadata") or {}).get("dataset_type") or "")
                    for case in merged_cases
                    if str((case.get("metadata") or {}).get("dataset_type") or "")
                }
            ),
            default=[],
        )
        search_query = st.text_input("キーワード検索", value="")
        board_columns = int(st.slider("ボード列数", min_value=1, max_value=3, value=2))
        board_limit = int(st.slider("表示件数", min_value=1, max_value=300, value=80))

    filtered_cases = _filter_cases(
        merged_cases,
        routes=route_filter,
        edited_filter=edited_filter,
        favorite_filter=favorite_filter,
        dataset_types=dataset_type_filter,
        query=search_query,
    )[:board_limit]

    _render_summary(merged_cases, filtered_cases, len(selected_logs))
    _render_custom_case_creator(state, state_path)
    _render_export_controls(merged_cases, filtered_cases)

    st.divider()
    st.subheader("Dashboard")
    st.caption("ボードをクリックすると会話編集画面に移動します。")
    if not filtered_cases:
        st.info("条件に一致するケースがありません。")
        return

    for row_cases in _chunk(filtered_cases, board_columns):
        columns = st.columns(board_columns)
        for col, case in zip(columns, row_cases, strict=False):
            with col:
                _render_dashboard_card(case, state, state_path)


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
    favorite_count = sum(
        1 for case in all_cases if bool((case.get("metadata") or {}).get("favorite"))
    )
    st.write(
        {
            "selected_logs": selected_logs,
            "total_cases": len(all_cases),
            "visible_cases": len(visible_cases),
            "edited_cases": edited_count,
            "favorite_cases": favorite_count,
            "route_counts": {k: v for k, v in route_counts.items() if k},
        }
    )


def _render_dashboard_card(case: dict[str, Any], state: dict[str, Any], state_path: Path) -> None:
    case = ensure_case_defaults(case)
    case_id = str(case.get("case_id"))
    widget_suffix = hashlib.sha1(case_id.encode("utf-8")).hexdigest()[:10]

    metadata = case.get("metadata") or {}
    output_block = case.get("output") or {}
    source = case.get("source") or {}
    question = initial_user_question(case)

    with st.container(border=True):
        with st.form(f"dashboard_card_{widget_suffix}", clear_on_submit=False):
            left, right = st.columns([0.75, 0.25])
            with left:
                st.markdown(f"**{question[:80] or '(no question)'}**")
                st.caption(f"case_id: `{case_id}`")
                st.caption(
                    f"route: `{output_block.get('predicted_route') or '-'}` / "
                    f"dataset: `{metadata.get('dataset_type') or '-'}`"
                )
                if metadata.get("favorite"):
                    st.caption("favorite: true")
                if source.get("assistant_timestamp"):
                    st.caption(f"time: {source['assistant_timestamp']}")
            with right:
                edited = st.checkbox(
                    "編集済み",
                    value=bool(metadata.get("edited")),
                    key=f"dash_edited_{widget_suffix}",
                )
                favorite = st.checkbox(
                    "favorite",
                    value=bool(metadata.get("favorite")),
                    key=f"dash_favorite_{widget_suffix}",
                )

            dataset_type = st.selectbox(
                "データセット種別",
                options=DATASET_TYPE_OPTIONS,
                index=_dataset_type_index(str(metadata.get("dataset_type") or "")),
                key=f"dash_dtype_{widget_suffix}",
            )

            action_col1, action_col2 = st.columns([0.45, 0.55])
            with action_col1:
                save_clicked = st.form_submit_button("更新")
            with action_col2:
                open_clicked = st.form_submit_button("会話を開く", type="primary")

        if save_clicked or open_clicked:
            metadata["edited"] = bool(edited)
            metadata["favorite"] = bool(favorite)
            metadata["dataset_type"] = dataset_type
            case["metadata"] = metadata
            upsert_case_in_state(case, state)
            save_workbench_state(state_path, state)

        if save_clicked:
            st.success(f"更新しました: {case_id}")

        if open_clicked:
            st.session_state[SELECTED_CASE_KEY] = case_id
            st.rerun()


def _render_conversation_editor(case: dict[str, Any], state: dict[str, Any], state_path: Path) -> None:
    case = ensure_case_defaults(case)
    case_id = str(case.get("case_id"))
    suffix = hashlib.sha1(case_id.encode("utf-8")).hexdigest()[:10]
    metadata = case.get("metadata") or {}
    labels = case.get("labels") or {}
    output_block = case.get("output") or {}
    source = case.get("source") or {}

    top_left, top_right = st.columns([0.65, 0.35])
    with top_left:
        if st.button("← ダッシュボードに戻る"):
            st.session_state[SELECTED_CASE_KEY] = None
            st.rerun()
    with top_right:
        st.caption(f"case_id: `{case_id}`")

    st.subheader("Conversation Editor")
    st.caption("`app.py` に近い会話表示で、メッセージを直接編集できます。")
    st.write(
        {
            "log_path": source.get("log_path"),
            "predicted_route": output_block.get("predicted_route"),
            "assistant_timestamp": source.get("assistant_timestamp"),
        }
    )

    conversation = case_to_conversation(case)
    conversation = normalize_conversation(conversation)

    with st.form(f"conversation_editor_{suffix}", clear_on_submit=False):
        head_left, head_right = st.columns([0.75, 0.25])
        with head_left:
            dataset_type = st.selectbox(
                "データセット種別",
                options=DATASET_TYPE_OPTIONS,
                index=_dataset_type_index(str(metadata.get("dataset_type") or "")),
                key=f"editor_dtype_{suffix}",
            )
            predicted_route = st.selectbox(
                "predicted_route",
                options=ROUTE_OPTIONS,
                index=_route_index(str(output_block.get("predicted_route") or "")),
                key=f"editor_pred_route_{suffix}",
            )
            expected_route = st.selectbox(
                "expected_route",
                options=ROUTE_OPTIONS,
                index=_route_index(str(labels.get("expected_route") or "")),
                key=f"editor_exp_route_{suffix}",
            )
            labeler = st.text_input(
                "labeler",
                value=str(labels.get("labeler") or ""),
                key=f"editor_labeler_{suffix}",
            )
            label_note = st.text_input(
                "label_note",
                value=str(labels.get("label_note") or ""),
                key=f"editor_label_note_{suffix}",
            )
            favorite_note = st.text_input(
                "favorite_note",
                value=str(metadata.get("favorite_note") or ""),
                key=f"editor_favorite_note_{suffix}",
            )
        with head_right:
            edited = st.checkbox(
                "編集済み",
                value=bool(metadata.get("edited")),
                key=f"editor_edited_{suffix}",
            )
            favorite = st.checkbox(
                "favorite",
                value=bool(metadata.get("favorite")),
                key=f"editor_favorite_{suffix}",
            )

        st.markdown("### 会話")
        if not conversation:
            st.info("会話が空です。下の追加入力から作成してください。")

        for index, message in enumerate(conversation):
            message_role = message["role"]
            with st.chat_message(message_role):
                row1, row2 = st.columns([0.7, 0.3])
                with row1:
                    st.selectbox(
                        f"role_{index}",
                        options=CHAT_ROLE_OPTIONS,
                        index=CHAT_ROLE_OPTIONS.index(message_role),
                        key=f"msg_role_{suffix}_{index}",
                    )
                with row2:
                    st.checkbox("削除", value=False, key=f"msg_delete_{suffix}_{index}")
                st.text_area(
                    f"message_{index}",
                    value=message["content"],
                    height=120,
                    key=f"msg_content_{suffix}_{index}",
                )

        st.markdown("### メッセージ追加")
        add_role = st.selectbox(
            "追加するrole",
            options=CHAT_ROLE_OPTIONS,
            index=0,
            key=f"add_role_{suffix}",
        )
        add_content = st.text_area(
            "追加するテキスト（空なら追加しない）",
            value="",
            height=100,
            key=f"add_content_{suffix}",
        )

        save_col, save_back_col = st.columns([0.45, 0.55])
        with save_col:
            save_clicked = st.form_submit_button("保存", type="primary")
        with save_back_col:
            save_and_back_clicked = st.form_submit_button("保存してボードへ戻る")

    if not (save_clicked or save_and_back_clicked):
        return

    delete_indexes: set[int] = set()
    editable_conversation: list[dict[str, str]] = []
    for index in range(len(conversation)):
        if st.session_state.get(f"msg_delete_{suffix}_{index}"):
            delete_indexes.add(index)
        role = st.session_state.get(f"msg_role_{suffix}_{index}", "user")
        content = str(st.session_state.get(f"msg_content_{suffix}_{index}", "")).strip()
        if role not in CHAT_ROLE_OPTIONS:
            role = "user"
        editable_conversation.append({"role": role, "content": content})

    rebuilt_conversation = delete_conversation_messages_by_index(editable_conversation, delete_indexes)

    add_text = str(add_content or "").strip()
    if add_text:
        rebuilt_conversation.append({"role": add_role, "content": add_text})

    if not rebuilt_conversation:
        st.error("会話が空になる保存はできません。")
        return

    case = apply_conversation_to_case(case, rebuilt_conversation)
    metadata = case.get("metadata") or {}
    labels = case.get("labels") or {}
    output_block = case.get("output") or {}

    metadata["edited"] = bool(edited)
    metadata["favorite"] = bool(favorite)
    metadata["favorite_note"] = favorite_note.strip() or None
    metadata["dataset_type"] = dataset_type
    output_block["predicted_route"] = predicted_route or ""
    labels["expected_route"] = expected_route or None
    labels["label_status"] = "labeled" if expected_route else "unlabeled"
    labels["labeler"] = labeler.strip() or None
    labels["label_note"] = label_note.strip() or None

    case["metadata"] = metadata
    case["labels"] = labels
    case["output"] = output_block

    upsert_case_in_state(case, state)
    save_workbench_state(state_path, state)
    _reset_conversation_delete_flags(suffix)

    if save_and_back_clicked:
        st.session_state[SELECTED_CASE_KEY] = None
        st.success(f"保存しました: {case_id}")
        st.rerun()

    st.success(f"保存しました: {case_id}")
    st.rerun()


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


def _filter_cases(
    cases: list[dict[str, Any]],
    *,
    routes: list[str],
    edited_filter: str,
    favorite_filter: str,
    dataset_types: list[str],
    query: str,
) -> list[dict[str, Any]]:
    normalized_query = query.strip().lower()
    selected_routes = set(routes)
    selected_dataset_types = set(dataset_types)

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

        favorite = bool(metadata.get("favorite"))
        if favorite_filter == "favorite_only" and not favorite:
            continue
        if favorite_filter == "non_favorite_only" and favorite:
            continue

        dataset_type = str(metadata.get("dataset_type") or "")
        if selected_dataset_types and dataset_type not in selected_dataset_types:
            continue

        if normalized_query:
            blob = _build_search_blob(case)
            if normalized_query not in blob:
                continue
        filtered.append(case)
    return filtered


def _build_search_blob(case: dict[str, Any]) -> str:
    output_block = case.get("output") or {}
    metadata = case.get("metadata") or {}
    labels = case.get("labels") or {}
    chunks = [
        initial_user_question(case),
        str(output_block.get("assistant_output") or ""),
        str(output_block.get("predicted_route") or ""),
        str(metadata.get("dataset_type") or ""),
        str(labels.get("expected_route") or ""),
    ]
    for item in case_to_conversation(case):
        chunks.append(item["content"])
    return "\n".join(chunks).lower()


def _dataset_type_index(current_value: str) -> int:
    return DATASET_TYPE_OPTIONS.index(current_value) if current_value in DATASET_TYPE_OPTIONS else 0


def _route_index(route_value: str) -> int:
    return ROUTE_OPTIONS.index(route_value) if route_value in ROUTE_OPTIONS else 0


def _chunk(items: list[dict[str, Any]], size: int) -> list[list[dict[str, Any]]]:
    if size <= 0:
        return [items]
    return [items[idx : idx + size] for idx in range(0, len(items), size)]


def _reset_conversation_delete_flags(suffix: str) -> None:
    delete_prefix = f"msg_delete_{suffix}_"
    for key in list(st.session_state.keys()):
        if key.startswith(delete_prefix):
            st.session_state[key] = False


if __name__ == "__main__":
    main()
