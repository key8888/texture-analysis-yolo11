# fine_tuning.py

from ultralytics import YOLO
from multiprocessing import freeze_support



if __name__ == '__main__':
    freeze_support()
    
    # 1. モデルの読み込み（事前学習済み重みを利用）
    model = YOLO("yolo11x-seg.pt")

    # 2. トレーニングの実行
    # ※ data.yaml のパスやエポック数、画像サイズなどはタスクに合わせて変更してください
    results = model.train(
        data="./fine/custom_dataset.yaml",  # data.yaml のファイルパス
        epochs=300,                 # エポック数
        imgsz=512,                  # 入力画像サイズ
        batch=16,                   # バッチサイズ（必要に応じて変更）
        lr0=0.005,                    # 初期学習率（例）
        device="cuda"
    )

    # 3. トレーニング結果の表示
    print("Training finished. Results:")
    print(results)
