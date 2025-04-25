import argparse
import random
import shutil
import subprocess
import sys

import numpy as np
import pandas as pd
import yaml
from pathlib import Path

from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow import keras as tfk
import os
import inference_pipeline
import api
import training_api

seed = 69

def main(args):
    if args.detection_only and args.classification_only:
        raise ValueError("--detection_only and --classification_only cannot be specified at the same time,"
                         " as they are mutually exclusive")
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

    if not args.classification_only:
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

    if not args.detection_only:
        best_model = None
        best_map50 = 0
        run_dir = os.path.join("runs", "detect")
        for train_dir in os.listdir(run_dir):
            results_path = os.path.join(run_dir, train_dir, "results.csv")
            df = pd.read_csv(results_path)
            best_row = df.loc[df['metrics/mAP50(B)'].idxmax()]
            map50 = best_row['metrics/mAP50(B)']
            if best_map50 < map50:
                best_map50 = map50
                best_model = os.path.join(run_dir, train_dir, "weights", "best.pt")

        if args.verbose:
            print("corrector training will be performed on predictions of model", best_model,
                  "which achieved map50", best_map50)

        original_argv = list(sys.argv)
        inference_args = f"inference_pipeline.py --input_folder {os.path.join(args.dataset, 'train', 'images')} " \
                         f"--model {best_model} --conf 0.01 --img_size {args.img_size} " \
                         f"--detection_only --write_conf --silent".split()
        sys.argv = list(inference_args)
        if args.verbose:
            print("Inferring on training dataset, may take a few seconds...")
        inference_args = inference_pipeline.parse_args()
        inference_pipeline.main(inference_args)

        api.warn_user_if_directory_exists("classify")
        os.makedirs(os.path.join("classify", "train"))
        if args.verbose:
            print("Storing true and false positives for posterior classification training...")
        training_api.save_tps_and_fps(os.path.join(args.dataset, 'train', 'images'), "output",
                                      os.path.join(args.dataset, 'train', 'labels'), os.path.join("classify", "train"))

        inference_args = f"inference_pipeline.py --input_folder {os.path.join(args.dataset, 'val', 'images')} " \
                         f"--model {best_model} --conf 0.01 --img_size {args.img_size} " \
                         f"--detection_only --write_conf --silent".split()
        sys.argv = list(inference_args)
        if args.verbose:
            print("Inferring on validation dataset, may take a few seconds...")
        inference_args = inference_pipeline.parse_args()
        inference_pipeline.main(inference_args)

        if args.verbose:
            print("Storing true and false positives for posterior classification validation...")
        training_api.save_tps_and_fps(os.path.join(args.dataset, 'val', 'images'), "output",
                                      os.path.join(args.dataset, 'val', 'labels'), os.path.join("classify", "val"))

        X_train, y_train = training_api.make_image_and_label_array(os.path.join("classify", "train"))
        X_val, y_val = training_api.make_image_and_label_array(os.path.join("classify", "val"))
        train_label_ratios = np.sum(y_train, axis=0) / len(y_train)
        val_label_ratios = np.sum(y_val, axis=0) / len(y_val)
        y_train = tfk.utils.to_categorical(y_train, num_classes=2)
        y_val = tfk.utils.to_categorical(y_val, num_classes=2)

        np.random.seed(seed)

        # Create a permutation of indices
        indices_train = np.random.permutation(len(X_train))
        X_train = X_train[indices_train]
        y_train = y_train[indices_train]
        indices_val = np.random.permutation(len(X_val))
        X_val = X_val[indices_val]
        y_val = y_val[indices_val]

        # Plot 10 random images
        # labels_txt = ["false positive", "true positive"]
        # random_indices = np.random.choice(np.arange(len(X_train)), size=10, replace=False)
        # training_api.plot_images(X_train[random_indices], [labels_txt[int(y_train[x])] for x in random_indices])
        # random_indices = np.random.choice(np.arange(len(X_val)), size=10, replace=False)
        # training_api.plot_images(X_val[random_indices], [labels_txt[int(y_val[x])] for x in random_indices])

        if args.verbose:
            print(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
            print(f"X_val length: {X_val.shape}, y_val shape: {y_val.shape}")
            print(f"Train Label Ratios (0: fp, 1: tp): {train_label_ratios}")
            print(f"Validation Label Ratios (0: fp, 1: tp): {val_label_ratios}")

        model = training_api.build_convnet()
        if args.verbose:
            model.summary()

        conv_history = model.fit(
            x=X_train,  # We need to apply the preprocessing thought for the ConvNeXt network, which is nothing
            y=y_train,
            # class_weight=class_weight_dict,
            batch_size=32,
            epochs=100,
            validation_data=(X_val, y_val),  # We need to apply the preprocessing thought for the ConvNeXt network
            callbacks=[
                tfk.callbacks.EarlyStopping(monitor='val_accuracy', mode='max', patience=10, restore_best_weights=True)]
        ).history

        # Plot the transfer learning and the fine-tuned ConvNeXt training histories
        plt.figure(figsize=(15, 5))
        plt.plot(conv_history['loss'], label='training', alpha=.3, color='#ff7f0e', linestyle='--')
        plt.plot(conv_history['val_loss'], label='validation', alpha=.8, color='#ff7f0e')

        plt.legend(loc='upper left')
        plt.title('Binary Crossentropy')
        plt.grid(alpha=.3)

        plt.figure(figsize=(15, 5))
        plt.plot(conv_history['accuracy'], label='training', alpha=.3, color='#ff7f0e', linestyle='--')
        plt.plot(conv_history['val_accuracy'], label='validation', alpha=.8, color='#ff7f0e')
        plt.legend(loc='upper left')
        plt.title('Accuracy')
        plt.grid(alpha=.3)

        plt.show()

        model.save("output.keras")

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
        "--detection_only",
        action="store_true",
        help="Disables corrector training, only trains detector"
    )
    parser.add_argument(
        "--classification_only",
        action="store_true",
        help="Disables detector training, only trains corrector"
    )
    # Parse arguments and run the main function
    args = parser.parse_args()
    main(args)