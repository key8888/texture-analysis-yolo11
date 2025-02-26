# evaluate.py

from ultralytics import YOLO

# 学習後の最良モデルを読み込み
model = YOLO("runs/segment/train2/weights/best.pt")

# 検証を実行
metrics = model.val(data="./fine/custom_dataset.yaml", imgsz=512)
print("Validation metrics:")
print(metrics)
