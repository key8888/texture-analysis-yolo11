import os
from ultralytics import YOLO
from pathlib import Path

def load_model(model_path):
    """
    モデルを読み込み、エラー発生時は例外を送出する。
    """
    try:
        model = YOLO(model_path)
        print("モデルの読み込みに成功しました。")
        return model
    except Exception as e:
        print(f"モデルの読み込み中にエラーが発生しました: {e}")
        raise

def get_target_class_index(model, target_class):
    """
    モデルに登録されているクラス名から対象クラスのインデックスを取得する。
    """
    for idx, name in model.names.items():
        if name == target_class:
            print(f"対象クラス '{target_class}' のインデックスは {idx} です。")
            return idx
    raise ValueError(f"対象クラス '{target_class}' がモデルに存在しません。")

def perform_inference(model, image_path, target_class_index, conf=0.7, iou=0.2, imgsz=512, device="cpu"):
    """
    画像ファイルの存在確認後、対象クラスのみ検出するように推論を実行する。
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"画像ファイルが存在しません: {image_path}")
    try:
        results = model.predict(
            source=image_path,      # 画像またはディレクトリ
            conf=conf,              # 信頼度の閾値
            iou=iou,                # NMS IoUの閾値
            device=device,          # 推論デバイス（CPUまたはGPU）
            imgsz=imgsz,            # 推論時のリサイズ
            classes=[target_class_index],  # 対象クラスのみ検出する
            agnostic_nms=False,
            show_labels=False,
            show_conf=True,
            max_det=1000,
            save=True,
            save_txt=True,
            save_conf=True,
            verbose=True,
            retina_masks=True
        )
        if not results:
            raise ValueError("推論結果が空です。")
        return results
    except Exception as e:
        print(f"推論中にエラーが発生しました: {e}")
        raise

def process_results(results):
    """
    推論結果からバウンディングボックス情報を抽出して表示する。
    """
    try:
        res = results[0]
    except IndexError:
        print("推論結果が見つかりません。")
        return

    if hasattr(res, "boxes") and res.boxes is not None:
        print(f"検出されたバウンディングボックス情報: {res.boxes}")
    else:
        print("バウンディングボックス情報が見つかりませんでした。")

def main():
    model_path = Path('runs\\independ')
    image_path = Path("YOLO_dataset_zip\\project-6-at-2025-03-23-20-14-00444e1f\\images")
    target_class = "y2o3"
          
    for img, mod in zip(image_path.iterdir(), model_path.iterdir()): # i want to iterate both the images and the model
        print(f'Image: {img}, Model: {mod}')
        try:
            model = load_model(mod / 'weights' / 'best.pt')
            target_index = get_target_class_index(model, target_class)
            results = perform_inference(model, img, target_index)
            # process_results(results)
        except Exception as e:
            print(f"エラーが発生しました: {e}")

if __name__ == "__main__":
    main()