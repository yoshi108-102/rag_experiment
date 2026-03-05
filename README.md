# Reflective Gate Chat (RAG Experiment)

作業や思考の振り返りを支援する「会話入口トリアージ型AIチャット」です。  
ユーザーの発話を `DEEPEN / CLARIFY / PARK / FINISH` に分類し、必要に応じて過去の知識（RAG）を参照して、近いテーマや新規性をデバッグ表示します。

## このシステムの概要

- 主な目的:
  - ユーザーに解決策を直接与えるのではなく、思考整理・言語化を促す
  - 発話をトリアージして、深掘りしすぎ/確認しすぎを避ける
  - 過去の蓄積知識と照合し、新規の気づきを `pending` として保存する
- 提供UI:
  - `app.py`: Streamlit UI（メイン）
  - `main.py`: CLI版（簡易）

## 全体アーキテクチャ

```text
User Input
  -> Gate (LLM分類: DEEPEN/CLARIFY/PARK/FINISH)
  -> Router (routeに応じた返答文を返す)
  -> (Streamlit時) RAG policy判定
       -> RAG検索（Embedding / n-gram fallback）
       -> 新規性判定
       -> pending保存（必要時）
  -> UI描画（応答 / Reasoning / RAG Debug）
  -> JSONLログ保存
```

### 主な責務

- `src/agents/gate.py`
  - Gate判定ユースケースの後方互換ファサード
- `src/chains/gate_classifier.py`
  - Gate判定（`GateDecision` のStructured Output）
  - Reasoning要約の取得と日本語翻訳呼び出し
  - `create_agent` の組み込みmiddlewareパイプラインで分類実行
- `src/middleware/prompt_middleware.py`
  - system prompt合成、メッセージ整形（画像含む）
  - `langchain.agents.middleware.dynamic_prompt` を使った prompt 注入
- `src/middleware/decision_guard.py`
  - ルールベース上書き（終了意図、苛立ち検知、CLARIFY完了判定）
- `src/routing/router.py`
  - `route` に応じた返答の返却（現状は `first_question` を返す薄い層）
- `src/chat_ui/turn_handler.py`
  - 1ターンの処理ユースケース（Gate -> Router -> RAG debug構築）
- `src/tools/rag_tools.py`
  - RAG機能のLangChain toolラッパ
- `src/chat_ui/rag_policy.py`
  - RAGのトリガ判定（境界ターン / ストリーク / クールダウン / 重複クエリ回避）
- `src/rag/*`
  - 知識ロード、検索、新規性判定、pending保存
- `src/core/chat_logging.py`
  - セッションログを `jsonl` 形式で保存

## 1ターン処理（Streamlit）

`app.py` では、入力1件ごとに `src/chat_ui/turn_handler.py` を呼び出して以下を実行します。

補足: `st.chat_input` は画像添付に対応しており、テキストのみ / 画像のみ / テキスト+画像 の送信が可能です。

1. `analyze_input()` で Gate判定（LLM + ルールベース補正）
2. `execute_route()` で返答文生成（現状は `first_question` を返す）
3. RAGポリシーで実行要否を判定
4. 実行時は知識検索 + 新規性判定 + `pending_reflections.jsonl` への保存
5. UIに以下を表示
   - 応答文
   - Reasoning（任意）
   - RAG検索結果 / 新規性判定（デバッグ）
   - ルーティング情報（デバッグ）
   - コンテキスト使用量（推定）と最新の実測 token usage（サイドバー）
6. 会話ログを `logs/chat_sessions/*.jsonl` に保存
7. Gate分類の生トレースを `logs/gate_agent_traces/*.jsonl` に保存（有効時）

## RAG（現在の実装）

- 知識ソース:
  - `datasets/consolidated/consolidated_knowledge.json`
- 検索:
  - 既定は OpenAI Embedding（`text-embedding-3-small`）
  - 失敗時は文字n-gram + 類似度ベース検索へフォールバック
- 新規性判定:
  - Top match score を閾値と比較して `is_novel` を判定
- 新規候補保存:
  - 新規性ありなら `datasets/pending/pending_reflections.jsonl` に重複排除して保存

## ディレクトリ構成（要点）

```text
.
├── app.py                    # Streamlitエントリポイント
├── main.py                   # CLIエントリポイント
├── prompts/                  # Gate用プロンプト、全体前提知識
├── datasets/                 # 原データ / 抽出結果 / 統合知識 / pending
├── logs/                     # チャットセッションログ
├── src/
│   ├── agents/               # Gate判定、Reasoning翻訳
│   ├── chains/               # LLMチェーン定義
│   ├── chat_ui/              # Streamlit向けUI/状態/ユースケース
│   ├── core/                 # 共通モデル、ロギング
│   ├── middleware/           # Prompt前処理/判定後ガードレール
│   ├── rag/                  # 検索・新規性判定・保存
│   ├── routing/              # route -> 応答
│   └── tools/                # LangChain tools
└── tests/                    # 単体テスト
```

補足: 責務分離の考え方は `docs/directory-structure.md` に整理されています。

## セットアップ（uv前提）

### 1. 依存関係のインストール

```bash
uv sync
```

### 2. 環境変数の設定

`.env.example` を参考に `.env` を作成してください。

```bash
cp .env.example .env
```

最低限必要な項目:

- `OPENAI_API_KEY`
- `RAG_EMBEDDING_MODEL`（任意。未指定時は `text-embedding-3-small`）
- `GATE_MODEL`（任意。未指定時は `gpt-5.2`）
- `GATE_CONTEXT_WINDOW_TOKENS`（任意。コンテキスト上限を明示したい場合に指定）
- `GATE_TRACE_LOG_ENABLED`（任意。`0/false/off/no`でGate生トレース保存を無効化）
- `REASONING_TRANSLATION_ENABLED`（任意。`1/true/on/yes` のときのみReasoning翻訳LLMを実行）
- `REASONING_TRANSLATION_MODEL`（任意。翻訳に使うモデル名。既定 `gpt-4o-mini`）
- `OVERALL_CONTEXT_MODE`（任意。`auto`/`always`/`off`。既定 `auto`）

## 実行方法

### Streamlit UI（推奨）

```bash
uv run streamlit run app.py
```

### CLI版

```bash
uv run python main.py
```

## テスト

ユニットテストは `tests/` 配下を対象に実行してください（ルートの `test_openai.py` は手動確認用）。

```bash
uv run pytest tests
```

## ログ分析（JSONL）

会話ログとGateトレースを集計するスクリプトを追加しています。

```bash
uv run python scripts/analyze_jsonl_logs.py
```

保存したい場合:

```bash
uv run python scripts/analyze_jsonl_logs.py --out logs/analysis/summary.json
```

## Evalデータセット作成（ログ由来）

`logs/chat_sessions/*.jsonl` から、評価用の1ターンケース（`user -> assistant`）を抽出して
`evals/cases/*.jsonl` を作成できます。出力ケースには `expected_route: null` が入るため、
その後に人手でラベル付けしてください。

```bash
uv run python scripts/build_eval_dataset_from_logs.py
```

主なオプション:

```bash
uv run python scripts/build_eval_dataset_from_logs.py \
  --max-cases 100 \
  --dedupe-mode user_and_route \
  --route-quota "CLARIFY=50,DEEPEN=20,FINISH=20,PARK=10" \
  --out evals/cases/route_eval_candidates_v1.jsonl
```

## EvalボードUI（Streamlit）

編集ワークベンチを使うと、履歴ログをダッシュボードで読み込み、ケースを会話UIで編集できます。

- ダッシュボードカードに「最初の疑問」を表示
- カード右上の `編集済み` チェックボックスで管理
- カードをクリックすると会話編集画面へ遷移
- 会話編集画面は `app.py` に近い `chat_message` 表示で、発話単位に編集可能
- `データセット種別` をケースごとに設定
- 新規ケースを自作して追加
- 編集済みケースだけをJSONL出力

起動方法（既存の `app.py` と同じマルチページアプリとして表示されます）:

```bash
uv run streamlit run app.py
```

サイドバーから `Eval Dataset Board` ページを選択してください。

## 補足（現状の仕様メモ）

- Gate判定は OpenAI Responses API + Structured Outputs を利用
- `prompts/gate_prompt.md` を読み込み、`prompts/overall.md` は `OVERALL_CONTEXT_MODE` に応じて注入
- `FINISH` / `PARK` は会話の区切りとして RAGバッファをクリア（RAG検索は実行しない）
- Streamlit UIでは RAG / Routing / Reasoning のデバッグ表示が有効
- Streamlit サイドバーにコンテキストウインドウ使用量（推定 + 最新API実測token usage）を表示
- Gate分類の深い実行ログ（request/response含む）は `logs/gate_agent_traces/*.jsonl` に保存
