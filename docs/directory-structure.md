# ディレクトリ構造メモ（ゆるDDD版）

このリポジトリは、厳密なDDDではなく「責務分離を優先した、ゆるいレイヤリング」を目指しています。

今回のリファクタでは、特に `app.py` に集中していた責務を `src/chat_ui/` に分割しました。

## 方針（ふんわりDDD）

- `domainっぽいもの`: 判定ルール・ポリシー（例: RAGをいつ走らせるか）
- `applicationっぽいもの`: 1ターンの処理を組み立てるユースケース
- `presentation`: Streamlit の表示・セッション状態
- `infrastructure`: LLM / RAG 実装、ファイルI/O など（既存 `src/agents`, `src/rag`）

厳密な境界よりも、「読んだ時に責任の所在が分かること」を優先しています。

## 現在の主要構造

```text
.
├── app.py                       # Streamlit の薄いエントリポイント（画面構成）
├── main.py                      # CLI エントリポイント
├── docs/
│   ├── directory-structure.md   # このファイル
│   └── current-workflow.md      # 現在の1ターン処理フロー
├── prompts/                     # LLM用プロンプト
├── src/
│   ├── agents/                  # LLM呼び出し・推論補助（Gate/翻訳）
│   ├── core/                    # 共通モデル（GateDecision）
│   ├── rag/                     # RAGパイプライン/検索/新規性判定/保存
│   ├── routing/                 # Route -> 応答のルーティング
│   └── chat_ui/                 # Streamlit向けの責務分離レイヤ
│       ├── constants.py         # UI/RAGの定数
│       ├── session_state.py     # Streamlit session_state の初期化/更新
│       ├── rag_policy.py        # RAG実行トリガのルール（ポリシー）
│       ├── turn_handler.py      # 1ターン処理の組み立て（ユースケース）
│       └── rendering.py         # Streamlit描画（履歴/デバッグ/RAGパネル）
└── tests/                       # 既存ロジックの単体テスト
```

## 役割分担の目安

- `app.py`:
  - 画面タイトル・入力欄・表示順序
  - 例外表示
  - 「何を表示するか」の最終オーケストレーション
- `src/chat_ui/turn_handler.py`:
  - Gate判定 -> Route実行 -> RAG判断の1ターン処理
  - UIから見た「会話処理ユースケース」
- `src/chat_ui/rag_policy.py`:
  - ストリーク / クールダウン / 境界ターンなどの判定ルール
  - ルール変更時に最初に触る場所
- `src/chat_ui/session_state.py`:
  - Streamlitセッションの初期化と更新
  - `app.py` から session の詳細を隠す
- `src/chat_ui/rendering.py`:
  - 表示ロジック（expanderやdebug表示）を集約

## 命名の考え方（今回の見直し）

- `pipeline.py` のような広すぎる名前は避ける
- `analyze_*`, `assess_*`, `search_*`, `store_*` のように「責務が動詞で読める」名前にする
- `loader.py`, `store.py` のような抽象名より、対象を含める（例: `knowledge_reader.py`, `pending_reflection_store.py`）

RAG配下の例:

- `reflection_context.py`: 反省文の文脈分析（検索 + 新規性判定 + pending保存）
- `record_search.py`: 類似レコード検索
- `novelty_rules.py`: 新規性判定ルール
- `knowledge_reader.py`: 統合済み知識の読み込み

## 今後の分離候補（必要になったら）

- `src/agents/gate.py` の promptロード / LLM呼び出し / レスポンスパースを分割
- `src/routing/router.py` の routeごとの処理をハンドラ分割（将来、PARK保存処理などが増える場合）
- `main.py` と `app.py` で共通の会話ユースケースをさらに共有
