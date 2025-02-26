from ultralytics import YOLO
import cv2
import matplotlib.pyplot as plt

# 1. モデルの読み込み
model = YOLO("runs/segment/train2/weights/best.pt")  # トレーニング済みの重みファイル

# 2. 入力画像のパスを指定
# image_path = 'fine/val/images/a3e1c24d-A231101_1_53Pa_350C_YBCO-STO_2_2_PlanView_13.bmp'
# image_path = 'fine/train/images/b0da50f5-A231101_1_53Pa_350C_YBCO-STO_2_2_PlanView_12.jpg'
image_path = "C:/Users/aottw/OneDrive/Pictures/shit.jpg"

# 3. 推論を実行
results = model.predict(source=image_path, save=True, imgsz=512)

# 4. 結果を表示
# 画像を読み込む
output_image_path = results[0].path  # 推論結果が保存された画像のパス
output_image = cv2.imread(output_image_path)
output_image = cv2.cvtColor(output_image, cv2.COLOR_BGR2RGB)  # BGRからRGBに変換

# Matplotlibで画像を表示
plt.figure(figsize=(10, 10))
plt.imshow(output_image)
plt.axis('off')
plt.title("Prediction Result")
plt.show()
