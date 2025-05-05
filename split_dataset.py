import os
import shutil
import argparse
import random
import api

seed = 69

def split_dataset(input_dir, output_dataset_dir, output_new_images_dir, split_ratio=0.9, seed=42):
    # Define subfolders
    subfolders = ['images', 'labels']

    # Set random seed for reproducibility
    random.seed(seed)

    # Create output directories
    for subfolder in subfolders:
        os.makedirs(os.path.join(output_dataset_dir, subfolder), exist_ok=True)
        os.makedirs(os.path.join(output_new_images_dir, subfolder), exist_ok=True)

    # List all image files
    images_path = os.path.join(input_dir, 'images')
    labels_path = os.path.join(input_dir, 'labels')
    all_images, _ = api.get_images_and_labels(images_path, labels_path)

    # Shuffle and split
    random.shuffle(all_images)
    split_idx = int(len(all_images) * split_ratio)
    dataset_images = all_images[:split_idx]
    new_images = all_images[split_idx:]

    print(f"Total images: {len(all_images)}")
    print(f"Training set size (dataset): {len(dataset_images)}")
    print(f"Validation set size (new_images): {len(new_images)}")

    # Helper function to copy paired files
    def copy_pair(image_list, destination_dir):
        for image_file in image_list:
            # Copy image
            src_image = os.path.join(input_dir, 'images', image_file)
            dst_image = os.path.join(destination_dir, 'images', image_file)
            shutil.copy2(src_image, dst_image)

            # Copy corresponding label
            label_file = os.path.splitext(image_file)[0] + '.txt'
            src_label = os.path.join(input_dir, 'labels', label_file)
            dst_label = os.path.join(destination_dir, 'labels', label_file)
            if os.path.exists(src_label):
                shutil.copy2(src_label, dst_label)
            else:
                print(f"Warning: No label found for {image_file}")

    # Copy files
    copy_pair(dataset_images, output_dataset_dir)
    copy_pair(new_images, output_new_images_dir)

    print("Dataset split complete!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dataset", type=str, required=True, help="Path to the input dataset (with images/ and labels/)")
    parser.add_argument("--output_dataset", type=str, default="dataset", help="Path to output 'dataset' folder (default: ./dataset)")
    parser.add_argument("--output_new_images", type=str, default="new_images", help="Path to output 'new_images' folder (default: ./new_images)")
    parser.add_argument("--dataset_size", type=int, default=None,
                        help="Output dataset size. When set, split ratio is ignored (default: agrees with split ratio)")
    parser.add_argument("--split_ratio", type=float, default=0.8, help="Split ratio (default: 0.8)")

    args = parser.parse_args()

    if args.dataset_size is not None:
        input_dataset_size = len(os.listdir(os.path.join(args.input_dataset, "labels")))
        args.split_ratio = args.dataset_size / input_dataset_size

    split_dataset(args.input_dataset, args.output_dataset, args.output_new_images, split_ratio=args.split_ratio, seed=seed)
