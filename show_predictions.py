import argparse
import os

import cv2
import api


def main(args):
    for i in os.listdir(args.images):
        image = cv2.imread(os.path.join(args.images, i))
        if args.predictions is not None:
            pred_file = os.path.join(args.predictions, i[:-4] + ".txt")
            pred_list = api.txt_to_tuple_list(pred_file) if os.path.exists(pred_file) else []
            if len(pred_list) > 0 and len(pred_list[0]) == 4:
                pred_list = [x + (1,) for x in pred_list]
            pred_list = [api.yolo_to_bbox(x, image.shape[1], image.shape[0]) for x in pred_list]
            pred_list = [x for x in pred_list if x[4] > args.min_conf]
            pred_list = api.remove_overlapping_regions_wrt_iou(pred_list, overlap_treshold=args.max_overlap)
            api.draw_bboxes(image, pred_list, color=(255, 0, 0), thickness=5, show_conf=not args.hide_conf)
        cv2.imshow(i, image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="python show_predictions.py --images my_image_folder "
                                                 "--predictions my_prediction_folder")

    parser.add_argument(
        "--images",
        type=str,
        required=True,
        help="Path to the images folder"
    )
    parser.add_argument(
        "--predictions",
        type=str,
        default=None,
        help="Path to the predictions folder (default: None, no bbox will be drawn)"
    )
    parser.add_argument(
        "--hide_conf",
        action="store_true",
        help="Hide confidence scores"
    )
    parser.add_argument(
        "--min_conf",
        type=float,
        default=0,
        help="Minimum confidence to show boxes (default: 0)"
    )
    parser.add_argument(
        "--max_overlap",
        type=float,
        default=1,
        help="Maximum overlap between detections (default: 1, which means no overlap threshold)"
    )
    # Parse arguments and run the main function
    args = parser.parse_args()
    main(args)