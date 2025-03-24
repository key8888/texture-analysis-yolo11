from ultralytics import YOLO
from multiprocessing import freeze_support
from pathlib import Path

def main():
    model_path = Path('runs/independ/no_spcdr_1/weights/best.pt')
    custom_dataset = Path('fine/no_spcdr_1/custom_dataset.yaml')

    model = YOLO(model_path)

    metrics = model.test(
        data=custom_dataset, # custom_dataset.yaml のファイルパス
        conf=0.3,            # 信頼度のしきい値
        iou=0.3,             # IoUのしきい値（評価用）
        save_json=True,      # 結果をJSON形式で保存
        verbose=True         # 詳細表示
    )

    # クラスごとの精度と再現率を表示
    for i, (p, r) in enumerate(zip(metrics.box.p, metrics.box.r)):
        print(f"Class {i} - Precision: {p:.4f}, Recall: {r:.4f}")

    # mAP@0.5とmAP@0.5:0.95を表示
    print(f"mAP@0.5: {metrics.box.map50:.4f}")
    print(f"mAP@0.5:0.95: {metrics.box.map:.4f}")

if __name__ == '__main__':
    freeze_support()
    main()
