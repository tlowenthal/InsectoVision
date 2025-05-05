import argparse
import os

import cv2
import numpy as np

import api
def main(args):
    nb_images = len(os.listdir(args.ground_truth))  # does not take into account empty boxes
    precisions = np.zeros(nb_images)
    recalls = np.zeros(nb_images)
    f1_scores = np.zeros(nb_images)
    map50s = np.zeros(nb_images)
    map50_95s = np.zeros(nb_images)

    # Get all image files that have matching label files
    image_extensions = (".jpg", ".jpeg", ".png")
    image_files = [
        f for f in os.listdir(args.images)
        if f.lower().endswith(image_extensions) and
           os.path.exists(os.path.join(args.ground_truth, os.path.splitext(f)[0] + ".txt"))
    ]

    for idx, i in enumerate(image_files):

        image = cv2.imread(os.path.join(args.images, i))

        pred_file = os.path.join(args.predictions, i[:-4] + ".txt")
        pred_list = api.txt_to_tuple_list(pred_file) if os.path.exists(pred_file) else []
        if len(pred_list) > 0 and len(pred_list[0]) == 4:
            pred_list = [x + (1,) for x in pred_list]
        pred_list = [api.yolo_to_bbox(x, image.shape[1], image.shape[0]) for x in pred_list]
        pred_list = [x for x in pred_list if x[4] > args.min_conf]
        pred_list = api.remove_overlapping_regions_wrt_iou(pred_list, overlap_treshold=args.max_overlap)

        gt_file = os.path.join(args.ground_truth, i[:-4] + ".txt")
        gt_list = api.txt_to_tuple_list(gt_file)
        gt_list = [x + (1,) for x in gt_list]
        gt_list = [api.yolo_to_bbox(x, image.shape[1], image.shape[0]) for x in gt_list]

        fp_list, metrics = api.evaluate_detections(pred_list, gt_list, 0.25)
        tp_list = [x for x in pred_list if x not in fp_list]

        if args.verbose:
            print("precision :", metrics["precision"])
            print("recall :", metrics["recall"])
            print("f1_score :", metrics["f1_score"])
            print("map@50 :", api.compute_map_50(pred_list, gt_list))
            print("map@50_95 :", api.compute_map_50_95(pred_list, gt_list))

        precisions[idx] = metrics["precision"]
        recalls[idx] = metrics["recall"]
        f1_scores[idx] = metrics["f1_score"]
        map50s[idx] = api.compute_map_50(pred_list, gt_list)
        map50_95s[idx] = api.compute_map_50_95(pred_list, gt_list)

    print("Average performance metrics over all images :")
    print("precision :", np.mean(precisions))
    print("recall :", np.mean(recalls))
    print("f1_score :", np.mean(f1_scores))
    print("map@50 :", np.mean(map50s))
    print("map@50_95 :", np.mean(map50_95s))

def parse_args():
    parser = argparse.ArgumentParser(description="python inference_pipeline.py --input_folder my_image_folder")

    parser.add_argument(
        "--images",
        type=str,
        required=True,
        help="Path to the images folder"
    )
    parser.add_argument(
        "--predictions",
        type=str,
        required=True,
        help="Path to the predictions folder"
    )
    parser.add_argument(
        "--ground_truth",
        type=str,
        required=True,
        help="Path to the labels folder"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose output"
    )
    parser.add_argument(
        "--min_conf",
        type=float,
        default=0,
        help="Minimum confidence threshold for predictions to be taken into account (default: 0)"
    )
    parser.add_argument(
        "--max_overlap",
        type=float,
        default=1,
        help="Maximum overlap between detections (default: 1, which means no overlap threshold)"
    )
    parser.add_argument(
        "--no_map",
        action="store_true",
        help="Set this flag when map50 and map50-95 are irrelevant metrics"
    )
    # Parse arguments and run the main function
    return parser.parse_args()

if __name__ == "__main__":

    # Parse arguments and run the main function
    args = parse_args()
    main(args)