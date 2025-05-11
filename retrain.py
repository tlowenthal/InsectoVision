import argparse
import os
import shutil
import sys
import random

import api
import training_pipeline

seed = 69


def main(args):
    api.warn_user_if_directory_exists("new_dataset", silent=args.silent)
    S, k = api.merge_datasets(args.dataset, args.new_images, "new_dataset", seed=seed)
    factor = k / (S + k)

    new_train_size = len(os.listdir(os.path.join("new_dataset", "train", "images")))
    new_val_size = len(os.listdir(os.path.join("new_dataset", "val", "images")))
    if args.verbose:
        print("Original dataset size:", S, "\nNumber of images to add:", k,
              "\nNew dataset size:", new_train_size + new_val_size,
              "\nNew validation ratio", new_val_size / (new_train_size + new_val_size))

    original_argv = list(sys.argv)
    new_epochs = max(int(args.original_epochs * factor), 3)
    new_steps = max(int(args.original_nb_steps * factor), 3)
    training_args = f"training_pipeline.py --dataset new_dataset --fine_tuning_steps {new_steps} " \
                     f"--model {args.model} --lr0 {args.original_lr0 * factor} --gpu {args.gpu} " \
                    f"--epochs {new_epochs} --batch_init {args.original_batch} " \
                    f"--replace_all --patience {max(new_epochs // 3, 2)} --no_split".split()
    if args.detection_only:
        training_args.append("--detection_only")
    if args.classification_only:
        training_args.append("--classification_only")
    if args.verbose:
        training_args.append("--verbose")
    sys.argv = list(training_args)
    training_args = training_pipeline.parse_args()
    training_pipeline.main(training_args)

    api.warn_user_if_file_exists("new_model.pt")
    shutil.copy2("output.pt", "new_model.pt")
    os.remove("output.pt")

    api.warn_user_if_file_exists("new_classifier.keras")
    shutil.copy2("output.keras", "new_classifier.keras")
    os.remove("output.keras")

    if (not (args.detection_only or args.classification_only)) and (not args.no_high_precision):

        training_args = f"training_pipeline.py --dataset new_dataset --fine_tuning_steps {new_steps} " \
                        f"--model new_model.pt --heatmap_extractor new_model.keras --lr0 {args.original_lr0 * factor} " \
                        f"--epochs {new_epochs} --batch_init {args.original_batch} " \
                        f"--replace_all --patience {max(new_epochs // 3, 2)} --no_split".split()
        if args.verbose:
            training_args.append("--verbose")
        sys.argv = list(training_args)
        training_args = training_pipeline.parse_args()
        training_pipeline.main(training_args)

        api.warn_user_if_file_exists("new_high_detection_model.pt")
        shutil.copy2("output.pt", "new_high_detection_model.pt")
        os.remove("output.pt")


def parse_args():
    parser = argparse.ArgumentParser(description="python training_pipeline.py --dataset my_dataset")

    # Add command-line arguments
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Path to the dataset, in standard yolo format (with subfolders images and labels)"
    )
    parser.add_argument(
        "--new_images",
        type=str,
        required=True,
        help="Path to the new images to add to the dataset, in standard yolo "
             "format (with subfolders images and labels)"
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Pretrained detection model to fine-tune on new samples"
    )
    parser.add_argument(
        "--gpu",
        type=str,
        default="mps",
        help="Gpu to use, the default is the macos standard (default: mps)"
    )
    parser.add_argument(
        "--original_lr0",
        type=float,
        default=0.01,
        help="Initial learning rate which was used to train the input model (default: 0.01)"
    )
    parser.add_argument(
        "--original_batch",
        type=int,
        default=16,
        help="Initial batch size which was used to train the input model (default: 16)"
    )
    parser.add_argument(
        "--original_epochs",
        type=int,
        default=20,
        help="Initial number of epochs which was used to train the input model (default: 20)"
    )
    parser.add_argument(
        "--original_nb_steps",
        type=int,
        default=10,
        help="Initial number of fine-tuning steps which was used to train the input model (default: 10)"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose output"
    )
    parser.add_argument(
        "--silent",
        action="store_true",
        help="Replace all existing directories without warning user"
    )
    parser.add_argument(
        "--detection_only",
        action="store_true",
        help="Disables corrector training, only trains detector"
    )
    parser.add_argument(
        "--classification_only",
        action="store_true",
        help="Disables detector training, only trains corrector"
    )
    parser.add_argument(
        "--no_high_precision",
        action="store_true",
        help="Disables high-precision low-recall model training"
    )

    return parser.parse_args()

if __name__ == "__main__":
    # Parse arguments and run the main function
    args = parse_args()
    main(args)