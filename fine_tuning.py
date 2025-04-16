from ultralytics import YOLO
from multiprocessing import freeze_support
from pathlib import Path
import time
import sys

def precheck_paths(fine_path: Path, runs_path: Path):
    # "fine" ディレクトリの存在チェック
    if not fine_path.exists() or not fine_path.is_dir():
        print(f"エラー: 指定されたディレクトリ {fine_path} は存在しません。")
        sys.exit(1)
    # "runs" ディレクトリが存在しない場合は作成する
    if not runs_path.exists():
        print(f"{runs_path} が存在しないため、ディレクトリを作成します。")
        exit(1)
        runs_path.mkdir(parents=True, exist_ok=True)

if __name__ == '__main__':
    freeze_support()  # マルチプロセシングを使用する場合のサポート
    
    # パスの定義
    fine = Path("data_for_training")
    runs = Path("runs")
    
    # パスチェック
    precheck_paths(fine, runs)
    
    # 高精度な時間計測開始
    start = time.perf_counter()
    
    # "data_for_training" 内の各サブディレクトリに対して学習を実行
    for spcdr in fine.iterdir():
        if spcdr.is_dir():
            print(f"開始: {spcdr.stem} のトレーニング")
            
            # 1. モデルの読み込み（事前学習済み重みの利用）
            model = YOLO("yolo11x.pt")
            
            # 2. トレーニングの実行
            results = model.train(
                data=spcdr / "custom_dataset.yaml",  # カスタムデータセットのファイルパス
                epochs=1000,                          # エポック数
                imgsz=512,                            # 入力画像サイズ
                batch=16,                             # バッチサイズ（タスクに応じて変更）
                lr0=0.005,                            # 初期学習率
                device="cuda",
                project=runs / "BoundingBox",         # モデルの保存先ディレクトリ
                name=spcdr.stem                       # 各サブディレクトリ名を保存フォルダ名として利用
            )
            
            print(f"{spcdr.stem} トレーニング終了")
    
    # 高精度な時間計測終了
    end = time.perf_counter()
    elapsed_time = end - start
    print(f"全処理時間: {elapsed_time:.2f}秒")
