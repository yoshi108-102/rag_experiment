import os
import json
import glob
from openai import OpenAI
from dotenv import load_dotenv

# 環境変数の読み込み (.envファイル等)
load_dotenv()
client = OpenAI()

def read_json_files(directory: str) -> list:
    """指定ディレクトリから全てのJSONファイルを読み込む"""
    json_data = []
    file_paths = glob.glob(os.path.join(directory, "*.json"))
    for path in file_paths:
        try:
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                # ソース元ファイル名を付与
                data['source_file'] = os.path.basename(path)
                json_data.append(data)
        except Exception as e:
            print(f"Error reading {path}: {e}")
    return json_data

def consolidate_knowledge(json_data_list: list) -> dict:
    """LLMを用いて散在するJSONデータを統合し、タグ付けを行う"""
    
    # すべてのJSONを一つの文字列にまとめる
    combined_json_str = json.dumps(json_data_list, ensure_ascii=False, indent=2)
    
    system_prompt = """
あなたは暗黙知統合のエキスパートです。
複数の作業ログ（JSON形式）から抽出された「観察事実」「仮説」「手法」「疑問」を分析し、
類似している内容を名寄せ（統合）し、検索しやすい形に構造化・タグ付けしてください。

【出力JSONスキーマ】
{
  "consolidated_knowledge": [
    {
      "topic": "統合された大テーマの名称（例: 太い棒の矯正方法）",
      "tags": ["対象:太い", "形状:弓なり", "動作:押す"],
      "summary_statement": "複数の入力を統合し抽象化した、このトピックの結論や傾向",
      "related_observations": ["統合元の具体的な発言や観察のリスト"],
      "methods": [
         {
           "description": "具体的な対処法",
           "applicable_when": "適用条件"
         }
      ],
      "open_questions": ["このトピックに関して未解決・検証が必要な疑問や仮説"]
    }
  ]
}

【統合のルール】
- 同じ対象（例：太い棒）や同じ作業（例：曲がりの確認）に関する項目は1つのトピックにまとめること
- 元の発言のニュアンス（related_observations）は失わないように残すこと
- 推論はせず、与えられたデータ群からのみタグ付けとまとめを行うこと
- 出力は必ず指定された形式のJSONのみであること
"""
    
    print("AIによる知識統合・名寄せを実行中...（時間がかかる場合があります）")
    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": combined_json_str}
            ],
            temperature=0.2
        )
        return json.loads(response.choices[0].message.content)
    except Exception as e:
        print(f"API Error during consolidation: {e}")
        return None

def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(base_dir, "output")
    consolidated_dir = os.path.join(base_dir, "consolidated")
    
    os.makedirs(consolidated_dir, exist_ok=True)
    
    # 1. 抽出済みJSONの読み込み
    extracted_data = read_json_files(output_dir)
    if not extracted_data:
        print("統合可能な抽出済みJSONファイルが見つかりません。先に抽出スクリプトを実行してください。")
        return
        
    print(f"{len(extracted_data)} 件のJSONファイルを結合対象として読み込みました。")
    
    # 2. LLMによる統合とタグ付け
    consolidated_data = consolidate_knowledge(extracted_data)
    
    if consolidated_data:
        # 3. 統合結果の保存
        out_path = os.path.join(consolidated_dir, "consolidated_knowledge.json")
        with open(out_path, 'w', encoding='utf-8') as f:
            json.dump(consolidated_data, f, ensure_ascii=False, indent=2)
        print(f"知識の統合・名寄せが完了しました。出力先: {out_path}")
    else:
        print("統合処理に失敗しました。")

if __name__ == "__main__":
    main()
