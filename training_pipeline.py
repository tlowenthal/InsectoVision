import argparse
import random
import shutil
import subprocess
import sys

import yaml
from pathlib import Path
from sklearn.model_selection import train_test_split
import os

seed = 69

def main(args):
    if args.lr_final == -1:
        args.lr_final = args.lr0 / 10
    if args.batch_decrease_rate == -1:
        args.batch_decrease_rate = (args.batch_init - args.batch_min) / (args.fine_tuning_steps - 1)

    # Configuration
    dataset_dir = args.dataset
    original_images_dir = os.path.join(dataset_dir, "images")
    original_labels_dir = os.path.join(dataset_dir, "labels")
    output_yaml_path = os.path.join(dataset_dir, "data.yaml")
    class_names = ["insect"]  # Replace with actual class names
    val_split = 0.2
    random.seed(seed)

    # Get all image files that have matching label files
    image_extensions = (".jpg", ".jpeg", ".png")
    image_files = [
        f for f in os.listdir(original_images_dir)
        if f.lower().endswith(image_extensions) and
           os.path.exists(os.path.join(original_labels_dir, os.path.splitext(f)[0] + ".txt"))
    ]

    # Split into train and val
    train_files, val_files = train_test_split(image_files, test_size=val_split, random_state=seed)

    # Define function to copy files
    def copy_files(file_list, split_type):
        images_target = os.path.join(dataset_dir, split_type, "images")
        labels_target = os.path.join(dataset_dir, split_type, "labels")
        os.makedirs(images_target, exist_ok=True)
        os.makedirs(labels_target, exist_ok=True)

        for filename in file_list:
            base = os.path.splitext(filename)[0]
            image_src = os.path.join(original_images_dir, filename)
            label_src = os.path.join(original_labels_dir, base + ".txt")

            image_dst = os.path.join(images_target, filename)
            label_dst = os.path.join(labels_target, base + ".txt")

            shutil.copy2(image_src, image_dst)
            shutil.copy2(label_src, label_dst)

    # Copy files to new structure
    copy_files(train_files, "train")
    copy_files(val_files, "val")

    # Get absolute paths for yaml
    train_images_abs = os.path.abspath(os.path.join(dataset_dir, "train", "images"))
    val_images_abs = os.path.abspath(os.path.join(dataset_dir, "val", "images"))

    # Write the YAML file
    data_yaml = {
        "train": train_images_abs,
        "val": val_images_abs,
        "nc": len(class_names),
        "names": class_names
    }

    with open(output_yaml_path, "w") as f:
        yaml.dump(data_yaml, f, sort_keys=False)

    if args.verbose:
        print(f"✅ Dataset prepared and YAML saved to: {output_yaml_path}")

    command = (
        f"python fine_tune_yolo.py --model {args.model} "
        f"--dataset {args.dataset} --epochs {args.epochs} --img_size {args.img_size} "
        f"--batch_init {args.batch_init} --batch_min {args.batch_min} --batch_decrease_rate {args.batch_decrease_rate} "
        f"--patience {args.patience} --fine_tuning_steps {args.fine_tuning_steps} "
        f"--lr0 {args.lr0} --lr_final {args.lr_final} --gpu {args.gpu}"
    )
    print(f"Running: {command}")
    try:
        subprocess.run(command, shell=True)
    except subprocess.CalledProcessError as e:
        print(f"Subprocess failed with return code {e.returncode}")
        sys.exit(1)  # Stops the main script with a non-zero exit code

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="python training_pipeline.py --dataset my_dataset")

    # Add command-line arguments
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Path to the dataset, in standard yolo format (with subfolders images and labels)"
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
        default=5,
        help="Number of fine-tuning runs, with a constant rate of layer "
             "unfreezing down to 0 frozen layers (default: 5)"
    )
    parser.add_argument(
        "--lr0",
        type=float,
        default=0.01,
        help="Initial learning rate (default: 0.01)"
    )
    parser.add_argument(
        "--lr_final",
        type=float,
        default=-1,
        help="Final run learning rate (default: lr0/100)"
    )
    parser.add_argument(
        "--batch_init",
        type=int,
        default=16,
        help="Initial batch size (default: 16)"
    )
    parser.add_argument(
        "--batch_min",
        type=int,
        default=8,
        help="Minimal batch size from which it will start plateauing (default: 8)"
    )
    parser.add_argument(
        "--batch_decrease_rate",
        type=float,
        default=-1,
        help="Rate at which batch size is reduced for each subsequent run "
             "(default: (batch_init - batch_min) / (fine_tuning_steps - 1))"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=20,
        help="Maximal number of epochs per run (default: 20)"
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=5,
        help="Patience fo early stopping (default: 5)"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="model/yolov8s.pt",
        help="Pretrained detection model to fine-tune on dataset (default: yolov8s.pt)"
    )
    parser.add_argument(
        "--img_size",
        type=int,
        default=640,
        help="Detector's input image size (default: 640)"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose output"
    )
    parser.add_argument(
        "--train_classifier",
        action="store_true",
        help="Enable corrector training"
    )
    # Parse arguments and run the main function
    args = parser.parse_args()
    main(args)