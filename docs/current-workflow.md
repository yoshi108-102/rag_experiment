# 現在のワークフロー（コード読解ガイド）

このドキュメントは、現行実装の「1ターン処理」をコード上で追うためのガイドです。

動的な可視化版: `docs/current-workflow-visualizer.html`

## 先に読むファイル順

1. `app.py`（Streamlitエントリ）
2. `src/chat_ui/turn_handler.py`（1ターンのオーケストレーション）
3. `src/agents/gate.py`（後方互換ファサード）
4. `src/chains/gate_classifier.py`（Gate分類本体）
5. `src/middleware/prompt_middleware.py`（プロンプト注入とメッセージ整形）
6. `src/middleware/decision_guard.py`（判定オーバーライド）
7. `src/routing/router.py`（routeに応じた返答）
8. `src/chat_ui/rag_policy.py`（RAG起動ルール）
9. `src/tools/rag_tools.py` -> `src/rag/reflection_context.py`（RAG実行本体）

## 全体フロー（Streamlit）

```text
User Input (text / image)
  -> app.py
  -> handle_user_turn(...) in src/chat_ui/turn_handler.py
     -> analyze_input(...) in src/agents/gate.py
        -> GateClassifierChain.classify(...) in src/chains/gate_classifier.py
           -> create_agent(..., middleware=[dynamic_prompt, wrap_model_call])
           -> GateDecision JSON を取得
           -> decision_guard で最終補正
     -> execute_route(...) in src/routing/router.py
     -> rag_policy 判定
     -> run_reflection_context_lookup(...) in src/tools/rag_tools.py
        -> analyze_reflection_context(...) in src/rag/reflection_context.py
  -> app.py で描画/ログ保存
```

## Gate分類の詳細

`GateClassifierChain.classify()` の中では、以下を実行します。

1. `build_chat_messages()` で `HumanMessage/AIMessage` 列を作成
2. `create_agent()` で軽量エージェントを作成（toolsは空）
3. middleware:
   - `gate_system_prompt_middleware` (`@dynamic_prompt`): system prompt を注入
   - `build_gate_invoke_middleware` (`@wrap_model_call`): `response_format` を付けてモデル実行
4. 最終 `AIMessage` から JSON を取り出し `GateDecision` に変換
5. `apply_decision_overrides()` で補正
   - 終了意図 (`FINISH`) 優先
   - 苛立ちシグナル (`PARK`) 退避
   - CLARIFY完了条件で `FINISH` へ移行

## RAG実行の詳細

`turn_handler._build_rag_debug()` で以下の順に判定します。

1. `should_run_rag()`（boundary-skip/cooldown/streak）
2. `build_buffered_idea_query()` で検索クエリ作成
3. `should_skip_same_query()` で重複・短文除外
4. `run_reflection_context_lookup()` 実行
5. `finalize_rag_run()` でメタ更新

RAG本体 (`analyze_reflection_context`) は以下を担当します。

- 知識レコード読み込み
- 埋め込み検索（失敗時n-gram fallback）
- 新規性判定
- 新規なら pending 保存

## セッション状態（Streamlit）

`src/chat_ui/session_state.py` で主に次を管理します。

- `messages`: 画面表示用履歴
- `llm_context`: Gate入力用履歴
- `idea_buffer`: RAGトリガ判定向けバッファ（`PARK`/`FINISH` でクリア）
- `rag_meta`: ターン数、前回RAG実行情報

## CLIフロー

`main.py` は `analyze_input()` と `execute_route()` を直接呼び、RAGデバッグ表示は持ちません。

## 変更時の着眼点

- 会話方針を変える: `prompts/gate_prompt.md`, `decision_guard.py`
- 判定実行の仕組みを変える: `gate_classifier.py`, `prompt_middleware.py`
- RAG発火条件を変える: `rag_policy.py`
- RAG検索/新規性ロジックを変える: `src/rag/`
- UI表示やログ項目を変える: `app.py`, `rendering.py`, `chat_logging.py`

## 関連テスト

- Gate: `tests/test_gate.py`
- Router: `tests/test_router.py`
- RAG: `tests/test_rag.py`
- ログ: `tests/test_chat_logging.py`
- メディア: `tests/test_chat_media.py`
