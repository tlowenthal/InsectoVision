import os
import random
import sys

import numpy as np
import api
import inference_pipeline
import training_api


def random_sample(dataset, _, __, sample_size, seed):
    images_dir = os.path.join(dataset, "images")
    labels_dir = os.path.join(dataset, "labels")
    images, _ = api.get_images_and_labels(images_dir, labels_dir)
    images = os.listdir(images_dir)
    images.sort()

    random_sample_from_image_names(images, sample_size, seed)


def random_sample_from_image_names(images, sample_size, seed):

    np.random.seed(seed)

    random_indices = np.random.choice(np.arange(len(images)), size=sample_size, replace=False)
    return random_indices


def uniform_partition(dataset, _, __, sample_size, seed):
    images_dir = os.path.join(dataset, "images")
    labels_dir = os.path.join(dataset, "labels")
    images, _ = api.get_images_and_labels(images_dir, labels_dir)
    images.sort()

    random.seed(seed)
    offset = random.randrange(len(images))
    step = len(images) / sample_size
    indices = [(round(i) + offset) % len(images) for i in np.arange(0, len(images), step)]
    return indices


def get_lowest_confidence_files(output_dir, n, aggregate):
    confidence_data = []

    for filename in os.listdir(output_dir):
        filepath = os.path.join(output_dir, filename)
        if not os.path.isfile(filepath):
            continue

        with open(filepath, 'r') as f:
            confidences = []
            for line in f:
                parts = line.strip().split()
                if len(parts) < 6:  # YOLO + confidence should be at least 6 elements
                    continue
                try:
                    confidence = float(parts[-1])
                    confidences.append(confidence)
                except ValueError:
                    continue

        if confidences:
            mean_conf = aggregate(confidences)
            confidence_data.append((filename, mean_conf))

    # Sort by mean confidence and return the n lowest
    confidence_data.sort(key=lambda x: x[1])
    return [filename for filename, _ in confidence_data[:n]]


def max_uncertainty(dataset, model, sample_size, aggregate):
    images_dir = os.path.join(dataset, "images")
    labels_dir = os.path.join(dataset, "labels")
    imgs, _ = api.get_images_and_labels(images_dir, labels_dir)
    imgs.sort()
    sys.argv = f"inference_pipeline.py --input_folder {images_dir} --model {model} " \
               f"--detection_only --silent --write_conf".split()
    args = inference_pipeline.parse_args()
    inference_pipeline.main(args)

    lowest_confidence_indices = []
    image_extensions = [".jpg", ".jpeg", ".png"]
    image_extensions.extend([ext.upper() for ext in image_extensions])
    filter_empty_boxes(dataset, "output")
    for i in get_lowest_confidence_files("output", sample_size, aggregate):
        for ext in image_extensions:
            img = i[:-4] + ext
            if img in imgs:
                lowest_confidence_indices.append(imgs.index(img))
                break

    return lowest_confidence_indices


def max_mean_uncertainty_sample(dataset, _, model, sample_size, __):
    return max_uncertainty(dataset, model, sample_size, np.mean)


def filter_empty_boxes(dataset, dir_to_filter):
    labels_dir = os.path.join(dataset, "labels")
    images_dir = os.path.join(dataset, "images")

    _, labels = api.get_images_and_labels(images_dir, labels_dir)
    for f in os.listdir(dir_to_filter):
        if f not in labels:
            os.remove(os.path.join(dir_to_filter, f))


def diverse_sample(dataset, original_dataset, model, sample_size, seed):
    # First select representative sample from the training set.
    # Diversity will then be computed wrt to that selection.
    original_images_dir = os.path.join(original_dataset, "images")
    original_labels_dir = os.path.join(original_dataset, "labels")
    original_images, original_labels = api.get_images_and_labels(original_images_dir, original_labels_dir)
    original_images.sort()
    original_labels.sort()
    representative_indices = training_api.make_representative_split(original_images_dir,
                                                                    original_labels_dir, sample_size, seed)
    avg_detections = [training_api.make_average_detection(os.path.join(original_images_dir, original_images[i]),
                                             os.path.join(original_labels_dir, original_labels[i]), seed=seed)
                      for i in representative_indices]
    avg_detections = np.asarray(avg_detections)
    original_representative_features = training_api.extract_features(avg_detections)

    # Infer on the new images, to get bounding boxes
    images_dir = os.path.join(dataset, "images")
    sys.argv = f"inference_pipeline.py --input_folder {images_dir} --model {model} " \
               f"--detection_only --silent --write_conf".split()
    args = inference_pipeline.parse_args()
    inference_pipeline.main(args)

    # Calculate diversity and make selection
    labels_dir = "output"
    filter_empty_boxes(dataset, labels_dir)
    return training_api.make_representative_split(images_dir, labels_dir, sample_size,
                                                  seed=seed, original_features=original_representative_features)


def test_sample(sampling_method, model, sample_name="test"):
    dataset = "whole_dataset"

    labels_dir = os.path.join(dataset, "labels")
    images_dir = os.path.join(dataset, "images")

    original_dataset = "uniform_partition"

    indices = sampling_method(dataset, original_dataset, model, 30, 69)
    api.make_set_from_indices(sample_name, images_dir, labels_dir, indices)

#test_sample(uniform_partition, None, "uniform_partition")
#test_sample(max_mean_uncertainty_sample, "final_3.pt", "max_uncertainty")
test_sample(diverse_sample, "final_3.pt", "diverse")
