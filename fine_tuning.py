# fine_tuning.py

from ultralytics import YOLO
from multiprocessing import freeze_support
from pathlib import Path
import time

fine = Path("fine")
runs = Path("runs")


for spcdr in fine.iterdir():
    # print(spcdr) # fine\no_spcdr_1
    # print(spcdr.stem) # no_spcdr_1
    
    if __name__ == '__main__':
        freeze_support()
        
        # 1. モデルの読み込み（事前学習済み重みを利用）
        model = YOLO("yolo11x-seg.pt")

        # 2. トレーニングの実行
        # ※ data.yaml のパスやエポック数、画像サイズなどはタスクに合わせて変更してください
        results = model.train(
            data=spcdr/"custom_dataset.yaml",  # custom_dataset.yaml のファイルパス
            epochs=1000,                 # エポック数
            imgsz=512,                  # 入力画像サイズ
            batch=16,                   # バッチサイズ（必要に応じて変更）
            lr0=0.005,                    # 初期学習率（例）
            device="cuda",
            project=runs/"BoundingBox",   # 保存場所
            name=spcdr.stem       # 保存フォルダ名
        )
        
        print(f"{spcdr.stem}トレーニング終了")

        # # 3. トレーニング結果の表示
        # print("Training finished. Results:")
        # print(results)
