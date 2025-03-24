import os
import cv2
from ultralytics import YOLO


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


def get_target_class_indices(model, target_classes):
    """
    モデルに登録されているクラス名から、対象クラス群のインデックスを取得する。
    対象クラスが見つからない場合は例外を送出する。
    """
    indices = []
    for target in target_classes:
        found = False
        for idx, name in model.names.items():
            if name == target:
                print(f"対象クラス '{target}' のインデックスは {idx} です。")
                indices.append(idx)
                found = True
                break
        if not found:
            raise ValueError(f"対象クラス '{target}' がモデルに存在しません。")
    return indices


def perform_inference(model, image_path, target_class_indices, conf=0.4, iou=0.5, imgsz=512, device="cpu"):
    """
    画像ファイルの存在確認後、対象クラスのみ検出するように推論を実行する。
    自動保存は無効にし、後処理で描画するため show_labels は False に設定。
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"画像ファイルが存在しません: {image_path}")
    try:
        results = model.predict(
            source=image_path,             # 画像またはディレクトリ
            conf=conf,                     # 信頼度の閾値
            # NMS IoU = 0.5 → 0.5以上重なってたら削除（けっこう厳しめ）
            # NMS IoU = 0.7 → 0.7以上重なってたら削除（ゆるめ）
            # NMS IoU = 0.3 → ちょっとでも重なったら削除（超厳しい）
            iou=iou,                       # NMS IoUの閾値
            device=device,                 # 推論デバイス（CPUまたはGPU）
            imgsz=imgsz,                   # 推論時リサイズ
            classes=target_class_indices,  # 対象クラス群のインデックスを指定
            agnostic_nms=False,
            show_labels=False,             # 自動描画は無効（後処理で描画）
            show_conf=True,
            max_det=1000,
            save=False,                    # 自動保存は無効
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


def process_results(results, image_path, custom_mapping):
    """
    推論結果から検出ボックス情報を取得し、元画像に描画する。
    ・各ボックスは緑色で描画し、
    ・出力されるラベルは、結果オブジェクト内の names 属性を custom_mapping に従って置換したものを使用する。
    ラベルは小さいフォントで、背景は黒色の半透明で描画する。
    """
    try:
        res = results[0]
        print(f"推論結果:{res}")
    except IndexError:
        print("推論結果が見つかりません。")
        return

    # 元画像の読み込み
    image = cv2.imread(image_path)
    if image is None:
        print("画像の読み込みに失敗しました。")
        return

    # 結果オブジェクトの names 属性を、custom_mapping に従って更新する
    if hasattr(res, 'names'):
        for idx, label in res.names.items():
            if label in custom_mapping:
                old = res.names[idx]
                res.names[idx] = custom_mapping[label]
                print(f"結果オブジェクト内のラベル '{old}' を '{res.names[idx]}' に置換しました。")

    # YOLOの結果から検出されたボックス情報を取得
    try:
        # ボックス情報は [x1, y1, x2, y2, conf, cls] の形式の numpy 配列
        boxes = res.boxes.data.cpu().numpy() if hasattr(
            res.boxes.data, "cpu") else res.boxes.data
    except Exception as e:
        print(f"検出結果の取得に失敗しました: {e}")
        return

    # 描画パラメータ
    box_color = (255, 0, 0)       # 緑色
    box_thickness = 1
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5              # 小さいフォントサイズ
    text_thickness = 1
    text_color = (255, 255, 255)  # 白色
    bg_color = (0, 0, 0)          # 黒色（背景）
    alpha = 0.5                 # 半透明

    for box in boxes:
        # 各ボックス情報： [x1, y1, x2, y2, conf, cls]
        x1, y1, x2, y2, conf, cls = box
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        # バウンディングボックスの描画
        cv2.rectangle(image, (x1, y1), (x2, y2), box_color, box_thickness)

        # 結果オブジェクトの names 属性から、対応するラベルを取得
        label_text = res.names[int(cls)]

        # テキストサイズの取得
        (text_width, text_height), baseline = cv2.getTextSize(
            label_text, font, font_scale, text_thickness)

        # テキスト背景の左上座標（バウンディングボックス上部、画像外に出ないよう調整）
        text_x = x1
        text_y = y1 - 10 if y1 - 10 > text_height + \
            baseline else y1 + text_height + baseline

        # テキスト背景用の矩形座標
        rect_x1 = text_x
        rect_y1 = text_y - text_height - baseline
        rect_x2 = text_x + text_width
        rect_y2 = text_y + baseline

        # 半透明の背景描画のため、対象領域のオーバーレイを作成しブレンディング
        overlay = image.copy()
        cv2.rectangle(overlay, (rect_x1, rect_y1),
                      (rect_x2, rect_y2), bg_color, -1)
        image = cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0)

        # テキストの描画
        cv2.putText(image, label_text, (text_x, text_y), font,
                    font_scale, text_color, text_thickness, cv2.LINE_AA)

    output_path = "predict/output_custom_y_600.png"
    cv2.imwrite(output_path, image)
    print(f"注釈付き画像を {output_path} に保存しました。")
    print(f"boxes: {boxes.shape}")


def main():
    model_path = "runs/segment/5d5_epochs_600/weights/best.pt"
    image_path = "fine/val/images/5d59b254-A231027_3_53Pa_250C_YBCO-STO_PlanView_85.bmp"
    # 同時に検出する対象クラス群
    target_classes = [
        # "y2o3_appleGreen",
        # "y2o3_green",
        "y2o3",
        # "y2o3_peppermintGreen"
    ]
    # 出力画像上で表示するためのカスタムマッピング
    custom_mapping = {
        # "y2o3_appleGreen": "yag",
        # "y2o3_green": "yg",
        "y2o3": "y",
        # "y2o3_peppermintGreen": "ypg"
    }

    try:
        model = load_model(model_path)
        # 対象クラス群のインデックスを取得
        target_indices = get_target_class_indices(model, target_classes)

        # 推論実行
        results = perform_inference(model, image_path, target_indices)
        # 結果オブジェクト内の names を custom_mapping に従って更新し、描画する
        process_results(results, image_path, custom_mapping)
    except Exception as e:
        print(f"エラーが発生しました: {e}")


if __name__ == "__main__":
    main()
