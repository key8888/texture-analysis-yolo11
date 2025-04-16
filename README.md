# texture-analysis-yolo11

YOLOv11 を用いたテクスチャ画像の物体検出・セグメンテーションモデルの構築および評価を行うプロジェクトです。主に実験的なファインチューニング、独立検証、カスタム予測などを目的としています。

📁 ディレクトリ構成

.
├── .venv/                      # 仮想環境（.gitignore推奨）
├── data_for_training/         # 学習用画像データ（10クラス）
├── runs/                      # 学習結果保存用（BoundingBox, segment 等）
├── verification/              # 検証用スクリプトなど
├── yolo11n.pt                 # 軽量YOLOモデル（検出）
├── yolo11x.pt                 # 高性能YOLOモデル（検出）
├── yolo11x-seg.pt             # セグメンテーション用YOLOモデル
├── fine_tuning.py             # ファインチューニングスクリプト
├── custom_model_predict.py    # 推論スクリプト（基本）
├── custom_model_predict_2.py  # 推論スクリプト（設定変更版）
├── val.py                     # モデル評価用スクリプト
├── independ_val_example.py    # 独立検証用の例
├── path_check.py              # パス確認用スクリプト
├── process_txt_file.py        # アノテーション整形スクリプト
├── test_cuda.py               # CUDA動作確認用
├── memo.txt                   # メモ・ログなど
├── .gitignore                 # 除外ファイル設定
└── README.md                  # このファイル

🛠️ セットアップ

# 仮想環境の作成と有効化
python -m venv .venv
source .venv/Scripts/activate  # Windows の場合

# ライブラリのインストール（例）
pip install torch opencv-python ultralytics

🏋️‍♂️ モデルのファインチューニング

python fine_tuning.py

data.yaml の設定に従って学習を実行。

結果は runs/ 以下に保存されます。

🔍 推論

python custom_model_predict.py

任意の画像に対して物体検出またはセグメンテーションを行います。

custom_model_predict_2.py は別設定での推論バージョンです。

📊 評価

python val.py

テスト画像を用いた評価を行います。

independ_val_example.py では、独立した検証パターンの例を示しています。

📝 補足

スクリプト名

説明

path_check.py

データパスの確認

process_txt_file.py

YOLOフォーマット整形

test_cuda.py

CUDAが有効か確認

memo.txt

実験記録や作業メモ

📮 ライセンス・連絡

このリポジトリは研究・学習用途を想定しています。

このプロジェクトは公開時、トレニンーグ画像を隠しています

商用利用や転載はご遠慮ください。

問題や提案がある場合は Issue または Pull Request でご連絡ください。

