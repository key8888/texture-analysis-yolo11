from pathlib import Path
import json

def read_polygon_from_txt(file_path) -> list[list[float]]:
    """
    YOLO形式のラベルファイルを読み込む
    :param file_path: ラベルファイルのパス
    :return: ラベル情報のリスト
    """
    with open(file_path, "r") as f:
        lines = f.readlines()
        # 前後の空白を削除し[2:-1]、floatに変換する
        return [list(map(float, line[2:-1].split(" "))) for line in lines]
    
def polygon_to_xyxy(polygons:list, image_width=512, image_height=512) -> list[tuple[float, float, float, float]]:
    """
    xyxy形式で出力される
    """
    result = []
    
    for polygon in polygons:
        # 2次元の座標を取得
        xs = polygon[::2]
        ys = polygon[1::2]
        
        # 座標を512x512の画像サイズに変換
        x_coords = [x * image_width for x in xs]
        y_coords = [y * image_height for y in ys]
        
        # 最小値と最大値を取得
        min_x = min(x_coords)
        min_y = min(y_coords)
        max_x = max(x_coords)
        max_y = max(y_coords)
        
        result.append((min_x, min_y, max_x, max_y))
    return result

def compute_iou(box1, box2) -> float:
    """
    box1, box2: タプル (x1, y1, x2, y2)
    """
    # 交差部分の座標
    xi1 = max(box1[0], box2[0])
    yi1 = max(box1[1], box2[1])
    xi2 = min(box1[2], box2[2])
    yi2 = min(box1[3], box2[3])

    # 交差部分の面積
    inter_width = max(0, xi2 - xi1)
    inter_height = max(0, yi2 - yi1)
    intersection = inter_width * inter_height

    # 各ボックスの面積
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])

    # IoUの計算
    union = box1_area + box2_area - intersection
    if union == 0:
        return 0
    iou = intersection / union
    return iou


def read_xyxy_from_json(json_path: Path) -> list[tuple[float, float, float, float]]:
    """
    :param json_path: jsonファイルのパス
    """
    with open(json_path, "r") as f:
        data = json.load(f)
    
    boxes = []
    for group in data:
        for obj in group:
            box = obj['box']
            boxes.append((box['x1'], box['y1'], box['x2'], box['y2']))
        
    return boxes

def compute_true_positive(pred_boxes:list, gt_boxes:list, iou_threshold=0.5) -> int:
    """
    :param pred_boxes: 予測ボックスのリスト
    :param gt_boxes: グラウンドトゥルースボックスのリスト
    :param iou_threshold: IoUの閾値
    :return: True Positiveの数
    """
    tp = 0
    for pred_box in pred_boxes:
        for gt_box in gt_boxes:
            if compute_iou(pred_box, gt_box) >= iou_threshold:
                tp += 1
                break  # 一つでも一致すればカウント
    
    return tp

# # 例
pred_boxes = read_xyxy_from_json(Path("predict/result_no_spcdr_1.json"))
# for i, box in enumerate(box_a):
#     print(f"box {i}: {box}")

gt_boxes = read_polygon_from_txt(Path("YOLO_dataset_zip/project-6-at-2025-03-23-20-14-00444e1f/labels/spcdr_1.txt"))
gt_boxes = polygon_to_xyxy(gt_boxes)
# for i, box in enumerate(polygon_to_xyxy(box_b)):
#     print(f"box {i}: {box}")

print(compute_true_positive(pred_boxes, gt_boxes, iou_threshold=0.5))
