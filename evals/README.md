# Evals Workflow

## 1. 候補ケース作成

```bash
uv run python scripts/build_eval_dataset_from_logs.py \
  --max-cases 100 \
  --dedupe-mode user_and_route \
  --route-quota "CLARIFY=50,DEEPEN=20,FINISH=20,PARK=10" \
  --out evals/cases/route_eval_candidates_v1.jsonl
```

## 2. 人手ラベル付け

各JSON行の `labels.expected_route` を埋める:

- `DEEPEN`
- `CLARIFY`
- `PARK`
- `FINISH`

更新する主要フィールド:

- `labels.expected_route`
- `labels.label_status` (`labeled`)
- `labels.labeler` (例: `yoshi`)
- `labels.label_note` (任意)

## 3. 運用の注意

- まずは候補ファイルを複製して作業する（例: `route_eval_labeled_v1.jsonl`）。
- `output.predicted_route` は参考値であり、ラベル時は正解として扱わない。
- ケースに不足情報がある場合は `label_note` に理由を残し、必要なら除外する。

## 4. UI編集（Streamlit）

`uv run streamlit run app.py` を起動し、サイドバーから `Eval Dataset Board` を選択すると、
ボード形式で編集できます。

- カードに最初の疑問を表示し、クリックで会話編集画面へ遷移
- カード右上の `編集済み` チェックで進捗管理
- `データセット種別` をカード単位で変更
- 会話編集画面（chat_message表示）で発話を直接編集
- 新規ケースを自作して追加
- `favorite` チェックで「うまくいったケース」を保存

## 5. favorite から下書き10件を増殖

favorite 付きケースを種に、人手修正前提の下書きケースを生成します。

```bash
uv run python scripts/generate_eval_drafts_from_favorites.py \
  --state-path evals/workbench/state.json \
  --count 10
```

出力:

- `evals/cases/favorite/favorite_snapshot_<batch_id>.jsonl`
- `evals/cases/generated/generated_drafts_<batch_id>.jsonl`

同時に、生成ケースは `state.json` へも保存されるため、Eval Board 上でそのまま人手編集できます。
