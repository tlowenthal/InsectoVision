import os
import shutil

import cv2
import tensorflow as tf
import numpy as np
from PIL import Image

def store_predictions(results):
    """
    Extracts predictions from YOLO model results and returns a list of tuples:
    (x_center, y_center, width, height, confidence), all normalized.
    """
    predictions = []

    for result in results:
        boxes = result.boxes
        for box in boxes:
            x, y, w, h = box.xywhn[0].tolist()     # normalized coordinates
            conf = box.conf[0].item()              # confidence
            predictions.append((x, y, w, h, conf))

    return predictions

def yolo_to_bbox(yolo_bbox, img_width, img_height):
    """ Convert YOLO format (x_center, y_center, width, height) to (x_min, y_min, x_max, y_max). """

    if len(yolo_bbox) == 4:
        yolo_bbox = yolo_bbox + (1,)
    x_center, y_center, width, height, conf = yolo_bbox
    x_min = (x_center - width / 2) * img_width
    x_max = (x_center + width / 2) * img_width
    y_min = (y_center - height / 2) * img_height
    y_max = (y_center + height / 2) * img_height
    return [x_min, y_min, x_max, y_max, conf]

def reverse_axis(box, img_width, img_height):

    x1, y1, x2, y2, conf = box
    return img_width - x2, img_height - y2, img_width - x1, img_height - y1, conf


def compute_iou(box1, box2):
    """ Compute Intersection over Union (IoU) between two bounding boxes. """
    x1, y1, x2, y2, _ = box1
    x1g, y1g, x2g, y2g, _ = box2

    xi1 = max(x1, x1g)
    yi1 = max(y1, y1g)
    xi2 = min(x2, x2g)
    yi2 = min(y2, y2g)

    inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)
    box1_area = (x2 - x1) * (y2 - y1)
    box2_area = (x2g - x1g) * (y2g - y1g)

    union_area = box1_area + box2_area - inter_area
    return inter_area / union_area if union_area > 0 else 0

def is_inside(box1, box2):
    if box1 is None or box2 is None:
        return False

    x1, y1, x2, y2, _ = box1
    x1g, y1g, x2g, y2g, _ = box2

    ret = True
    ret &= (x1 > x1g and x2 < x2g)
    ret &= (y1 > y1g and y2 < y2g)

    return ret

def is_inside_and_centered(box1, box2):
    if not is_inside(box1, box2):
        return False

    x1, y1, x2, y2, conf = box2
    width = x2 - x1
    height = y2 - y1

    shift_right = x1 + width/2, y1, x2 + width/2, y2, conf
    shift_left = x1 - width/2, y1, x2 - width/2, y2, conf
    shift_up = x1, y1 + height/2, x2, y2 + height/2, conf
    shift_down = x1, y1 - height / 2, x2, y2 - height / 2, conf

    ret = True
    ret &= compute_iou(box1, shift_right) > 0
    ret &= compute_iou(box1, shift_left) > 0
    ret &= compute_iou(box1, shift_up) > 0
    ret &= compute_iou(box1, shift_down) > 0
    return ret

def is_largely_contained(box1, box2, threshold=0.5):

    if box1 is None or box2 is None:
        return False

    x1, y1, x2, y2, _ = box1
    x1g, y1g, x2g, y2g, _ = box2

    xi1 = max(x1, x1g)
    yi1 = max(y1, y1g)
    xi2 = min(x2, x2g)
    yi2 = min(y2, y2g)

    inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)
    box1_area = (x2 - x1) * (y2 - y1)
    box2_area = (x2g - x1g) * (y2g - y1g)
    return inter_area/box1_area > threshold or inter_area/box2_area > threshold

LABEL_BIAS = False
def evaluate_detections(pred_boxes, gt_boxes, iou_threshold=0.5, iosa_threshold=0.7):
    """ Compute precision, recall, and IoU-based accuracy. """
    tp, fp, fn = 0, 0, 0
    matched_preds = set()

    pred_boxes = sorted(pred_boxes, key=lambda x: x[4], reverse=True)
    tps = []
    fps = []

    for gt in gt_boxes:
        best_iou = 0
        best_match = None
        best_box = None
        for i, pred in enumerate(pred_boxes):
            iou = compute_iou(pred, gt)
            if iou > best_iou:
                best_iou = iou
                best_match = i
                best_box = pred

        bias_condition = LABEL_BIAS and is_largely_contained(best_box, gt, iosa_threshold)
        if (best_iou >= iou_threshold or bias_condition) and best_match not in matched_preds:
            tp += 1
            matched_preds.add(best_match)
            tps.append(1)
            fps.append(0)
        else:
            tps.append(0)
            fps.append(1)

    tps = np.cumsum(tps)
    fps = np.cumsum(fps)

    recalls = tps / len(gt_boxes)
    precisions = tps / (tps + fps + 1e-6)

    # Interpolate precision (COCO-style 101-point interpolation optional)
    ap = 0.0
    for t in np.linspace(0, 1, 101):
        p = precisions[recalls >= t]
        ap += max(p) if p.size else 0
    ap /= 101

    fp_list = [pred_boxes[i] for i in range(len(pred_boxes)) if i not in matched_preds]
    fp = len(fp_list)
    fn = len(gt_boxes) - len(matched_preds)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 1
    recall = tp / (tp + fn) if (tp + fn) > 0 else 1
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    return fp_list, {"precision": precision, "recall": recall, "f1_score": f1_score, "ap": ap}

def compute_map_50_95(pred_list, gt_list):
    aps = []
    for iou in np.arange(0.5, 1.0, 0.05):
        _, metrics = evaluate_detections(pred_list, gt_list, iou_threshold=iou)
        ap = metrics["ap"]
        aps.append(ap)
    return np.mean(aps)

def compute_map_50(pred_list, gt_list):
    _, metrics = evaluate_detections(pred_list, gt_list, iou_threshold=0.5)
    ap = metrics["ap"]
    return np.mean(ap)

def txt_to_tuple_list(file_name):
    ret_list = []
    with open(file_name, mode='r', newline='') as file:
        for line in file.readlines():
            row = line.split()
            if row[0][0] <= '9':
                ret_list.append(tuple([float(x) for x in row[1:]]))
    return ret_list

def draw_bboxes(image, bboxes, color=(0, 255, 0), thickness=2, show_conf=False):
    """ Draw bounding boxes on an image and display it. """

    for (x_min, y_min, x_max, y_max, conf) in bboxes:
        cv2.rectangle(image, (int(x_min), int(y_min)), (int(x_max), int(y_max)), color, thickness)

        if show_conf:

            # Prepare the confidence score text
            label = f"{conf:.2f}"

            # Choose font and size
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 2
            font_thickness = 4

            # Get the size of the text box
            (text_width, text_height), _ = cv2.getTextSize(label, font, font_scale, font_thickness)

            # Draw background rectangle for text (optional, improves readability)
            cv2.rectangle(image,
                          (int(x_min), int(y_min) - text_height - 8),
                          (int(x_min) + text_width, int(y_min)),
                          color,
                          cv2.FILLED)

            # Put the text above the top-left corner of the bounding box
            cv2.putText(image, label,
                        (int(x_min), int(y_min) - 6),
                        font, font_scale,
                        (0, 0, 0), font_thickness, lineType=cv2.LINE_AA)

def resize_image_opencv(image_path, output_size=(2016, 1216)):
    """ Resize an image to the given dimensions using OpenCV. """
    image = cv2.imread(image_path)  # Load image
    resized_image = cv2.resize(image, output_size, interpolation=cv2.INTER_LINEAR)
    return resized_image

def remove_overlapping_regions(bboxes, overlap_treshold=0.1):
    ret_list = []
    bboxes = sorted(bboxes, key=lambda x: x[4])
    for i in range(len(bboxes)):
        overlap_found = False
        for j in range(i+1, len(bboxes)):
            if is_largely_contained(bboxes[i], bboxes[j]):
            #if compute_iou(bboxes[i], bboxes[j]) > 0: #or is_largely_contained(bboxes[i], bboxes[j]):
                overlap_found = True
                break
        if not overlap_found:
            ret_list.append(bboxes[i])
    return ret_list

def filter_bboxes_zscore(bboxes, threshold=5):
    """ Remove bounding boxes with extreme areas using Z-score. """
    areas = np.array([(x_max - x_min) * (y_max - y_min) for x_min, y_min, x_max, y_max, _ in bboxes])

    mean_area = np.mean(areas)
    std_area = np.std(areas)

    filtered_bboxes = [
        b for b, area in zip(bboxes, areas) if abs((area - mean_area) / std_area) < threshold
    ]

    return filtered_bboxes

def save_regions(image, boxes, output_path, output_length, resize=None):

    for i, region in enumerate(boxes):
        x_min, y_min, x_max, y_max, _ = map(int, region)
        cropped_region = image[y_min:y_max, x_min:x_max]
        if resize is not None:
            cropped_region = tf.image.resize_with_pad(cropped_region, resize[0], resize[1])
            cropped_region = np.asarray(cropped_region, dtype=np.uint8)
        cv2.imwrite(output_path + "/i" + str(output_length + i) + ".jpg", cropped_region)
    return output_length + len(boxes)

def save_yolo_format(bboxes, image_size, output_txt_path, class_id=0, write_conf=False):
    """
    Saves bounding boxes in YOLO format to a .txt file.

    Parameters:
        bboxes: list of tuples (x_min, y_min, x_max, y_max, conf)
        image_path: path to the original image (for dimensions)
        output_txt_path: path to the .txt file to write
        class_id: default class id to assign to all boxes
    """

    # Get image dimensions
    img_width, img_height = image_size

    with open(output_txt_path, "w") as f:
        for bbox in bboxes:
            x_min, y_min, x_max, y_max, conf = bbox

            # Convert to YOLO format
            x_center = (x_min + x_max) / 2.0
            y_center = (y_min + y_max) / 2.0
            width = x_max - x_min
            height = y_max - y_min

            # Normalize
            x_center /= img_width
            y_center /= img_height
            width /= img_width
            height /= img_height

            # Write to file
            line_to_write = f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}"
            if write_conf:
                line_to_write += f" {conf:.6f}"
            f.write(line_to_write + "\n")


import numpy as np


def get_bbox_class_probs(pred_list, saliency_map, threshold=0.5):
    class_probs = []

    for (x_min, y_min, x_max, y_max, conf) in pred_list:
        # Convert to int in case coordinates are floats
        x_min, y_min, x_max, y_max = map(int, [x_min, y_min, x_max, y_max])

        # Clip to image boundaries
        x_min = max(0, x_min)
        y_min = max(0, y_min)
        x_max = min(saliency_map.shape[1], x_max)
        y_max = min(saliency_map.shape[0], y_max)

        # Crop saliency region
        region = saliency_map[y_min:y_max, x_min:x_max]

        avg_saliency = np.mean(region)
        label = 1 if avg_saliency >= threshold else 0
        class_probs.append(label)

    return class_probs


def warn_user_if_directory_exists(dir, silent=False):
    if os.path.exists(dir):
        if not silent:
            ans = input(f"{dir} folder already exists, do you wish to replace (r) or cancel (c) ?\n")
            while not (ans == 'r' or ans == 'c'):
                input("Invalid response, choose between replace (r) or cancel (c)\n")
            if ans == 'c':
                exit()
            else:
                shutil.rmtree(dir)
        else:
            shutil.rmtree(dir)
    os.makedirs(dir)

def warn_user_if_file_exists(file, silent=False):
    if os.path.exists(file):
        if not silent:
            ans = input(f"{file} file already exists, do you wish to replace (r) or cancel (c) ?\n")
            while not (ans == 'r' or ans == 'c'):
                input("Invalid response, choose between replace (r) or cancel (c)\n")
            if ans == 'c':
                exit()
            else:
                os.remove(file)
        else:
            os.remove(file)