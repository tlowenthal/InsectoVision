import argparse
import os
import shutil
import sys
import random

import api
import training_pipeline

seed = 69

def copy_folder(src_folder, dst_folder):
    os.makedirs(dst_folder, exist_ok=True)
    for filename in os.listdir(src_folder):
        src_file = os.path.join(src_folder, filename)
        dst_file = os.path.join(dst_folder, filename)
        if os.path.isfile(src_file):
            shutil.copy2(src_file, dst_file)

def shuffle_data(images, labels, seed):
    random.seed(seed)

    shuffled_images = list(images)
    shuffled_labels = list(labels)
    shuffled_images.sort()
    shuffled_labels.sort()
    indices = list(range(len(shuffled_images)))
    random.shuffle(indices)
    shuffled_images = [shuffled_images[i] for i in indices]
    shuffled_labels = [shuffled_labels[i] for i in indices]

    return shuffled_images, shuffled_labels

def merge_datasets(original_dataset_dir, new_dir, output_dir, new_split_ratio=0.8, seed=seed):
    random.seed(seed)

    train_path_src_images = os.path.join(original_dataset_dir, "train", "images")
    train_path_src_labels = os.path.join(original_dataset_dir, "train", "labels")
    val_path_src_images = os.path.join(original_dataset_dir, "val", "images")
    val_path_src_labels = os.path.join(original_dataset_dir, "val", "labels")

    train_path_dst_images = os.path.join(output_dir, "train", "images")
    train_path_dst_labels = os.path.join(output_dir, "train", "labels")
    val_path_dst_images = os.path.join(output_dir, "val", "images")
    val_path_dst_labels = os.path.join(output_dir, "val", "labels")

    new_images_dir = os.path.join(new_dir, "images")
    new_labels_dir = os.path.join(new_dir, "labels")

    copy_folder(train_path_src_images, train_path_dst_images)
    copy_folder(train_path_src_labels, train_path_dst_labels)
    copy_folder(val_path_src_images, val_path_dst_images)
    copy_folder(val_path_src_labels, val_path_dst_labels)

    train_length_src = len(os.listdir(train_path_src_images))
    val_length_src = len(os.listdir(val_path_src_images))
    S = train_length_src + val_length_src

    new_images, new_labels = api.get_images_and_labels(new_images_dir, new_labels_dir)
    k = len(new_labels)

    new_val_size = int(new_split_ratio * (S + k))
    k_val = new_val_size - val_length_src

    images_to_add, labels_to_add = shuffle_data(new_images, new_labels, seed)

    for i in range(k_val):
        image_name = images_to_add[i]
        label_name = labels_to_add[i]
        shutil.copy2(os.path.join(new_images_dir, image_name),
                     os.path.join(val_path_dst_images, image_name))
        shutil.copy2(os.path.join(new_labels_dir, label_name),
                     os.path.join(val_path_dst_labels, label_name))

    for i in range(k_val, k):
        image_name = images_to_add[i]
        label_name = labels_to_add[i]
        shutil.copy2(os.path.join(new_images_dir, image_name),
                     os.path.join(train_path_dst_images, image_name))
        shutil.copy2(os.path.join(new_labels_dir, label_name),
                     os.path.join(train_path_dst_labels, label_name))

    return S, k


def main(args):
    api.warn_user_if_directory_exists("new_dataset")
    S, k = merge_datasets(args.dataset, args.new_images, "new_dataset", seed=seed)
    factor = S / (S + k)

    new_train_size = len(os.listdir(os.path.join("new_dataset", "train")))
    new_val_size = len(os.listdir(os.path.join("new_dataset", "val")))
    if args.verbose:
        print("Original dataset size:", S, "\nNumber of images to add:", k,
              "\nNew dataset size:", new_train_size + new_val_size,
              "\nNew validation ratio", new_val_size / (new_train_size + new_val_size))

    original_argv = list(sys.argv)
    new_epochs = max(int(args.original_epochs * factor), 3)
    new_steps = max(int(args.original_nb_steps * factor), 3)
    training_args = f"training_pipeline.py --dataset new_dataset --fine_tuning_steps {new_steps} " \
                     f"--model {args.model} --lr0 {args.original_lr0 * factor} " \
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

    if not (args.detection_only or args.classification_only):

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
        default=3,
        help="Initial number of fine-tuning steps which was used to train the input model (default: 10)"
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

    return parser.parse_args()

if __name__ == "__main__":
    # Parse arguments and run the main function
    args = parse_args()
    main(args)