import argparse
import os

import api
import sys
import training_pipeline

seed = 69


def main(args):

    S, k = api.merge_datasets(args.dataset, args.new_images, args.output_dataset, seed=seed)
    if args.verbose:
        new_train_size = len(os.listdir(os.path.join("new_dataset", "train", "images")))
        new_val_size = len(os.listdir(os.path.join("new_dataset", "val", "images")))
        if args.verbose:
            print("Original dataset size:", S, "\nNumber of images to add:", k,
                  "\nNew dataset size:", new_train_size + new_val_size,
                  "\nNew validation ratio", new_val_size / (new_train_size + new_val_size))
            print("\n")

    sys.argv = f"training_pipeline.py --dataset new_dataset --gpu {args.gpu} --no_split " \
               f"--verbose --replace_all --fine_tuning_steps {args.fine_tuning_steps}".split()
    training_args = training_pipeline.parse_args()
    training_pipeline.main(training_args)
    api.warn_user_if_file_exists(args.output_model + ".pt", silent=True)
    os.rename("output.pt", args.output_model + ".pt")
    api.warn_user_if_file_exists(args.output_model + ".keras", silent=True)
    os.rename("output.keras", args.output_model + ".keras")


def parse_args():
    parser = argparse.ArgumentParser(description="python merge_and_retrain.py --dataset my_dataset "
                                                 "--new_images new_labelled_images")

    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Path to the original dataset, in standard yolo format (with subfolders images and labels)"
    )
    parser.add_argument(
        "--new_images",
        type=str,
        required=True,
        help="Path to the new labelled images, in standard yolo format (with subfolders images and labels)"
    )
    parser.add_argument(
        "--gpu",
        type=str,
        default="mps",
        help="Gpu to use, the default is the macos standard (default: mps)"
    )
    parser.add_argument(
        "--fine_tuning_steps",
        type=int,
        default=10,
        help="Number of fine-tuning runs, with a constant rate of layer "
             "unfreezing down to 0 frozen layers (default: 10)"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose output"
    )
    parser.add_argument(
        "--output_dataset",
        type=str,
        default="new_dataset",
        help="Path to the output dataset where the merging of the original dataset and the new images "
             "will be stored (default: new_dataset)"
    )
    parser.add_argument(
        "--output_model",
        type=str,
        default="new_model",
        help="Name of the resulting models, with extensions .pt and .keras (default: new_model, "
             "giving new_model.pt and new_model.keras)"
    )

    return parser.parse_args()


if __name__ == "__main__":
    # Parse arguments and run the main function
    args = parse_args()
    main(args)