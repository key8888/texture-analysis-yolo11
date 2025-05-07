from ultralytics import YOLO
from pathlib import Path
import json

images = Path("YOLO_dataset_zip/project-6-at-2025-03-23-20-14-00444e1f/images")

for img_path in images.iterdir():
    # モデルの読み込み
    print(f"使用画像：{img_path}, 使用モデル：no_{img_path.stem}/weights/best.pt")
    
    model = YOLO(Path(f"runs/BoundingBox/no_{img_path.stem}/weights/best.pt"))
    results = model.predict(img_path,
                save=True,  # 結果の保存
                save_txt=False,  # テキストファイルとして保存する
                save_conf=False,  # テキストファイル信頼度情報がを書き込む
                conf=0.7,
                iou=0.2,
                device='cpu',
                )

    json_list = []

    for r in results:
        json_list.append(json.loads(r.to_json()))
        
    with open(Path(f"predict/result_no_{img_path.stem}.json"), "w") as f:
        json.dump(json_list, f, indent=2)

# 一枚のみの時は、以下のようにしても良い
# for r in results:
#     json_data = r.to_json()
#     with open(Path("predict/result.json"), "w") as f:
#         f.write(json_data)
