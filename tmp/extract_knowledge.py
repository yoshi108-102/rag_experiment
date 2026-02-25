import os
import json
import glob
from pypdf import PdfReader
from openai import OpenAI
from dotenv import load_dotenv

# 環境変数の読み込み (.envファイル等)
load_dotenv()

# OpenAIクライアントの初期化 (OPENAI_API_KEY が設定されている前提)
# ※必要に応じて、LangChainやGoogle Generative AI等のSDKに変更可能です。
client = OpenAI()

def read_text_file(filepath: str) -> str:
    with open(filepath, 'r', encoding='utf-8') as f:
        return f.read()

def extract_text_from_pdf(pdf_path: str) -> str:
    """PDFからテキストを抽出する"""
    text = ""
    try:
        reader = PdfReader(pdf_path)
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
    except Exception as e:
        print(f"Error reading {pdf_path}: {e}")
    return text

def extract_knowledge(system_prompt: str, user_prompt_template: str, text: str) -> dict:
    """LLM呼び出しで知識を抽出する"""
    
    # 抽出対象のテキストを区切り文字で明確に示す
    user_message = f"{user_prompt_template}\n\n========== 抽出対象テキスト ==========\n{text}\n==================================="
    
    try:
        response = client.chat.completions.create(
            model="gpt-4o", # 抽出タスクは知能の高いモデル推奨
            response_format={"type": "json_object"}, # 出力を確実にJSONパースするためにJSONモード指定
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message}
            ],
            temperature=0.1 # 抽出・構造化タスクではハルシネーションを防ぐため低い温度を設定
        )
        result = response.choices[0].message.content
        return json.loads(result)
    except Exception as e:
        print(f"API Error: {e}")
        return None

def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    system_prompt_path = os.path.join(base_dir, "system_prompt.txt")
    user_prompt_path = os.path.join(base_dir, "user_prompt.txt")
    data_dir = os.path.join(base_dir, "data")
    output_dir = os.path.join(base_dir, "output")
    
    # 出力先ディレクトリの作成
    os.makedirs(output_dir, exist_ok=True)
    
    # プロンプトの読み込み
    system_prompt = read_text_file(system_prompt_path)
    
    # OpenAIのJSONモードを機能させるため、システムにもJSONに関する記述が必要な場合があるため補足
    if "JSON" not in system_prompt.upper():
        system_prompt += "\nOutput must be valid JSON."
        
    user_prompt_template = read_text_file(user_prompt_path)
    
    # 対象のPDFファイル一覧取得
    pdf_files = glob.glob(os.path.join(data_dir, "*.pdf"))
    if not pdf_files:
        print(f"'{data_dir}' ディレクトリにPDFファイルが見つかりません。")
        return
        
    print(f"合計 {len(pdf_files)} 個のPDFファイルを処理します...")
    
    # コンテキスト肥大化を防ぐため、1ファイルずつループ処理
    for pdf_path in sorted(pdf_files):
        filename = os.path.basename(pdf_path)
        print(f"\n--- 処理開始: {filename} ---")
        
        # 1. テキスト抽出
        print("1. PDF抽出中...")
        pdf_text = extract_text_from_pdf(pdf_path)
        if not pdf_text.strip():
            print(f"警告: {filename} からテキストを抽出できませんでした。スキップします。")
            continue
            
        print(f"--> {len(pdf_text)} 文字抽出完了")
        
        # ※ファイルサイズ(文字数)が巨大な場合は、ここでさらにチャンク分割(RecursiveCharacterTextSplitter等)
        # を行うロジックを挟むと精度が上がります。今回は1つのPDFそのまま送ります。
        
        # 2. LLMで情報抽出
        print("2. LLMによるJSON抽出実行中...")
        extracted_data = extract_knowledge(system_prompt, user_prompt_template, pdf_text)
        
        if extracted_data:
            # 3. ファイルごとの状態として保存
            output_filename = filename.replace('.pdf', '_extracted.json')
            output_path = os.path.join(output_dir, output_filename)
            
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(extracted_data, f, ensure_ascii=False, indent=2)
                
            print(f"3. 抽出結果を保存完了 -> {output_filename}")
        else:
            print(f"抽出失敗: {filename}")

if __name__ == "__main__":
    main()
