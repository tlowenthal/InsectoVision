import argparse
import os

import cv2
import numpy as np

import api


def main(args):
    nb_images = len(os.listdir(args.images))
    precisions = np.zeros(nb_images)
    recalls = np.zeros(nb_images)
    f1_scores = np.zeros(nb_images)
    map50s = np.zeros(nb_images)
    map50_95s = np.zeros(nb_images)

    for idx, i in enumerate(os.listdir(args.images)):

        image = cv2.imread(os.path.join(args.images, i))

        pred_list = api.txt_to_tuple_list(os.path.join(args.predictions, i[:-4] + ".txt"))
        if len(pred_list) > 0 and len(pred_list[0]) == 4:
            pred_list = [x + (1,) for x in pred_list]
        pred_list = [api.yolo_to_bbox(x, image.shape[1], image.shape[0]) for x in pred_list]

        gt_list = api.txt_to_tuple_list(os.path.join(args.ground_truth, i[:-4] + ".txt"))
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

if __name__ == "__main__":
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
    # Parse arguments and run the main function
    args = parser.parse_args()
    main(args)