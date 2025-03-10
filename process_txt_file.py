import os
from pathlib import Path

def process_file(file_path, output_folder):
    with open(file_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()
    
    processed_lines = []
    for line in lines:
        parts = line.strip().split()
        if parts:
            try:
                first_int = int(parts[0])
                if first_int in {3, 4, 5}:  # 3, 4, 5 の場合
                    parts[0] = '5'  # すべて5に統一
                processed_lines.append(' '.join(parts))
            except ValueError:
                continue  # 整数でない場合はスキップ
    
    os.makedirs(output_folder, exist_ok=True)
    output_file_path = output_folder / file_path.name
    with open(output_file_path, 'w', encoding='utf-8') as output_file:
        output_file.write('\n'.join(processed_lines) + '\n')

def process_all_files(input_folder):
    input_path = Path(input_folder)
    output_folder = input_path / 'output'
    txt_files = list(input_path.glob("*.txt"))
    
    for txt_file in txt_files:
        process_file(txt_file, output_folder)
    
    print(f"処理完了: {len(txt_files)} 個のファイルを処理しました。")

if __name__ == "__main__":
    input_folder = "./hello"
    process_all_files(input_folder)
