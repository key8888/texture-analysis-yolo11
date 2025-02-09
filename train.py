from ultralytics import YOLO

model = YOLO("yolo11x-seg.pt")

model.train(data="fine/custom_dataset.yaml", imgsz=512, device=0, batch=8, epochs=100)