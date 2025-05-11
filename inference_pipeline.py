import warnings
warnings.filterwarnings("ignore")

import argparse
import copy
import os
import shutil

import cv2
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
from ultralytics import YOLO
import api
import tensorflow as tf
from tensorflow import keras as tfk
import time
import training_api

def main(args):
    # Example usage of the parsed arguments

    if args.model.endswith(".pt"):
        to_be_ensembled = [args.model]
    else:
        to_be_ensembled = [os.path.join(args.model, x) for x in os.listdir(args.model) if x.endswith(".pt")]
        if len(to_be_ensembled) == 0:
            raise ValueError("Directory to ensemble does not contain .pt models")
    to_be_ensembled = [YOLO(x) for x in to_be_ensembled]

    api.warn_user_if_directory_exists("output", silent=args.silent)

    if args.high_precision:
        args.detection_only = True

    for image_file in os.listdir(args.input_folder):

        # if image_file != "Classified[]be-rbins-ent-belgian-microlepidoptera[]cosmopterigidae[]box-01[]a2.jpg":
        #     continue

        start = time.time()

        image_path = os.path.join(args.input_folder, image_file)
        image = Image.open(image_path)
        image_size = image.size

        if args.high_precision:
            img_array = np.array(image, dtype=np.float32)  # Convert to numpy array
            factor = np.max(img_array.shape) / 2016 if np.max(img_array.shape) > 2016 else 1
            img_array = tf.image.resize(img_array, (int(img_array.shape[0] / factor), int(img_array.shape[1] / factor)),
                                        method=tf.image.ResizeMethod.BILINEAR)

            #fcnn = tfk.models.load_model("fcnn.keras")
            fcnn = api.make_fcnn(args.classifier)
            preds = fcnn.predict(np.expand_dims(img_array, axis=0), verbose=args.verbose)
            preds = np.squeeze(preds)
            heatmap = tf.image.resize(np.expand_dims(preds[:, :, 1], axis=-1), (img_array.shape[0], img_array.shape[1]),
                                      method=tf.image.ResizeMethod.BILINEAR)
            heatmap = np.uint8(heatmap * 255)  # Scale to [0,255]
            heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)  # Apply color map
            heatmap_RGB = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
            alpha = 0.1
            image = cv2.addWeighted(np.uint8(img_array), 1 - alpha, heatmap_RGB, alpha,0)  # Blend images

        cv_image = cv2.imread(image_path)  # Load image
        cv_image_rgb = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
        pred_list = []
        for model in to_be_ensembled:
            results = model.predict(source=image, conf=args.conf, imgsz=args.img_size, iou=args.max_overlap,
                                    max_det=1000, verbose=args.verbose)
            pred = api.store_predictions(results)
            pred = [api.yolo_to_bbox(x, image_size[0], image_size[1]) for x in pred]
            pred = [x for x in pred if x[-1] > args.conf]
            pred = api.remove_overlapping_regions(pred)
            pred = api.filter_bboxes_zscore(pred)
            pred_list.extend(pred)
        pred_list = api.remove_overlapping_regions(pred_list)
        old_list = list(pred_list)

        if not args.detection_only and len(pred_list) > 0:
            classifier_name = args.classifier
            classifier = tfk.models.load_model(classifier_name)

            resized_pred_regions = []
            if args.resize_mode == "pad":
                for region in pred_list:
                    x_min, y_min, x_max, y_max, _ = map(int, region)
                    cropped_region = cv_image_rgb[y_min:y_max, x_min:x_max].astype(np.float32)
                    cropped_region = tf.image.resize_with_pad(cropped_region, 256, 256)
                    resized_pred_regions.append(cropped_region)
                resized_pred_regions = np.asarray(resized_pred_regions, dtype=np.float32)
            elif args.resize_mode == "bilinear":
                cv_image_rgb = cv_image_rgb.astype(np.float32)
                normalized_pred_list = []
                for region in pred_list:
                    x_min, y_min, x_max, y_max, _ = region
                    normalized_pred_list.append((y_min / image_size[1], x_min / image_size[0],
                                                y_max / image_size[1], x_max / image_size[0]))
                resized_pred_regions = tf.image.crop_and_resize(
                    image=tf.expand_dims(cv_image_rgb, axis=0),  # [1, H, W, C]
                    boxes=normalized_pred_list,  # [N, 4] in normalized coords [y1, x1, y2, x2]
                    box_indices=tf.zeros(len(normalized_pred_list), dtype=tf.int32),  # image index for each box
                    crop_size=(256, 256)
                )
            else:
                raise ValueError("Invalid resizing mode")
            # training_api.plot_images(resized_pred_regions.numpy(), [f"Pred {i}" for i in range(1, len(resized_pred_regions) + 1)])
            predictions = np.argmax(classifier.predict(resized_pred_regions, verbose=args.verbose), axis=-1) if len(
                resized_pred_regions) > 0 else np.array([])
            # print(model.predict(resized_pred_regions, verbose=0).astype(np.float64))
            indices = list(np.where(predictions == 1)[0]) if len(predictions) > 0 else []
            pred_list = [old_list[i] for i in range(len(old_list)) if i in indices]

        api.save_yolo_format(pred_list, image_size,
                             os.path.join("output", image_file[:-4] + ".txt"), write_conf=args.write_conf)

        end = time.time()
        if not args.silent:
            print(f"Time elapsed: {end - start:.4f} seconds")

        # cv_image = cv2.imread(os.path.join(args.input_folder, image_file))  # Load image
        # api.draw_bboxes(cv_image, pred_list, (255, 0, 0), 5)
        # api.draw_bboxes(cv_image, [x for x in old_list if x not in pred_list], (0, 255, 0), 5)
        # cv2.imshow("Image with Bounding Boxes", cv_image)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()


def parse_args():
    parser = argparse.ArgumentParser(description="python inference_pipeline.py --input_folder my_image_folder")

    # Add command-line arguments
    parser.add_argument(
        "--conf",
        type=float,
        default=0.125,
        help="Confidence threshold (default: 0.001)"
    )
    parser.add_argument(
        "--max_overlap",
        type=float,
        default=0.25,
        help="Maximum overlap between detections (default: 0.25)"
    )
    parser.add_argument(
        "--model",
        type=str,
        default=os.path.join("model", "final_23.pt"),
        help="Path to detection model (default: final_23.pt, in the model directory)"
    )
    parser.add_argument(
        "--classifier",
        type=str,
        default=os.path.join("model", "final_23.keras"),
        help="Path to posterior classifier (default: final_23.keras, in the model directory)"
    )
    parser.add_argument(
        "--resize_mode",
        type=str,
        default="pad",
        help="Resizing mode that was used in the training of the classifier. If you can afford "
             "a bit more overhead, training with mode \'pad\' will give greater accuracy. "
             "(default: \'pad\', faster but less accurate alternative : \'bilinear\'.)"
    )
    parser.add_argument(
        "--img_size",
        type=int,
        default=640,
        help="Detector's input image size (default: 640)"
    )
    parser.add_argument(
        "--input_folder",
        type=str,
        required=True,
        help="Path to the input folder"
    )
    parser.add_argument(
        "--high_precision",
        action="store_true",
        help="Use the high-precision low-recall model"
    )
    parser.add_argument(
        "--detection_only",
        action="store_true",
        help="Use detector only, and no classifier"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose output"
    )
    parser.add_argument(
        "--write_conf",
        action="store_true",
        help="Add confidence for each bounding box prediction in the output txt files"
    )
    parser.add_argument(
        "--silent",
        action="store_true",
        help="Nothing printed in stdout"
    )

    return parser.parse_args()


if __name__ == "__main__":
    # Parse arguments and run the main function
    args = parse_args()
    main(args)