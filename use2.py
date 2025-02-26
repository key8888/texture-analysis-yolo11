from ultralytics import YOLO

# モデルの読み込み
model = YOLO("runs/segment/train/weights/best.pt") 

# 入力画像のパス
image_path = 'fine/val/images/a3e1c24d-A231101_1_53Pa_350C_YBCO-STO_2_2_PlanView_13.bmp'

# 推論の実行（ラベルを非表示に設定）
results = model.predict(
    source=image_path,  # 画像 or ディレクトリなど
    conf=0.65,                  # confidence threshold
    iou=0.5,                   # NMS IoU threshold
    device="cpu",             # GPU推論（環境による）
    imgsz=512,                 # 推論時リサイズ
    agnostic_nms=False,        # クラスに依存したNMS
    show_labels=False,          # ラベル表示
    show_conf=True,            # スコア表示
    max_det=1000,              # 最大検出数
    save=True,                 # 結果画像を保存
    save_txt=True,             # バウンディングボックス座標をtxt保存
    save_conf=True,             # confidenceもtxtに含める
    verbose=True,
    retina_masks=True
)

res = results[0]

# print(f"result[0]: {res}")
print(f"検出されたバウンディングボックス情報 {res.boxes}")