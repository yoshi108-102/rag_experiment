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

- カード右上の `編集済み` チェックで進捗管理
- `データセット種別` をカード単位で変更
- 会話内容を直接編集
- 新規ケースを自作して追加
