import argparse
import os

import cv2
import api


def main(args):
    for i in os.listdir(args.images):
        image = cv2.imread(os.path.join(args.images, i))
        pred_list = api.txt_to_tuple_list(os.path.join(args.predictions, i[:-4] + ".txt"))
        pred_list = [x + (1,) for x in pred_list]
        pred_list = [api.yolo_to_bbox(x, image.shape[1], image.shape[0]) for x in pred_list]
        api.draw_bboxes(image, pred_list, (255, 0, 0), 5)
        cv2.imshow("Image with Bounding Boxes", image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

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

    # Parse arguments and run the main function
    args = parser.parse_args()
    main(args)