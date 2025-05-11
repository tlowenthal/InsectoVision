import os

import cv2

import random
import sys
from sklearn.metrics import pairwise_distances

import numpy as np
import api
import inference_pipeline
import training_api


def random_sample(dataset, _, __, sample_size, seed):
    images_dir = os.path.join(dataset, "images")
    images = api.get_images(images_dir)

    return random_sample_from_image_names(images, sample_size, seed)


def random_sample_from_image_names(images, sample_size, seed):

    np.random.seed(seed)

    random_indices = np.random.choice(np.arange(len(images)), size=sample_size, replace=False)
    return random_indices


def uniform_partition(dataset, _, __, sample_size, seed):
    images_dir = os.path.join(dataset, "images")
    images = api.get_images(images_dir)

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
    imgs = api.get_images(images_dir)
    sys.argv = f"inference_pipeline.py --input_folder {images_dir} --model {model} " \
               f"--detection_only --silent --write_conf".split()
    args = inference_pipeline.parse_args()
    inference_pipeline.main(args)

    lowest_confidence_indices = []
    image_extensions = [".jpg", ".jpeg", ".png"]
    image_extensions.extend([ext.upper() for ext in image_extensions])
    # filter_empty_boxes(dataset, "output")
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
    all_images = api.get_images(images_dir)
    valid_images, _ = api.get_images_and_labels(images_dir, labels_dir)
    # filter_empty_boxes(dataset, labels_dir)
    valid_indices = training_api.make_representative_split(images_dir, labels_dir, sample_size,
                                                  seed=seed, original_features=original_representative_features)
    return api.convert_valid_to_real_indices(valid_indices, valid_images, all_images)


def supervised_sample(dataset, original_dataset, model, sample_size, seed, max_to_consider=500, supervised_ratio=0.75):
    random.seed(seed)


    original_val_images_dir = os.path.join(original_dataset, "val", "images")
    original_val_labels_dir = os.path.join(original_dataset, "val", "labels")
    sys.argv = f"inference_pipeline.py --input_folder {original_val_images_dir} --model {model} " \
               f"--detection_only --silent --write_conf".split()
    args = inference_pipeline.parse_args()
    inference_pipeline.main(args)

    original_val_images, original_val_labels = api.get_images_and_labels(original_val_images_dir,
                                                                         original_val_labels_dir)
    original_val_images.sort()
    original_val_labels.sort()
    original_val_size = len(original_val_images)
    f1_scores = np.zeros(original_val_size)
    for idx, i in enumerate(original_val_images):

        image = cv2.imread(os.path.join(original_val_images_dir, i))

        pred_file = os.path.join("output", i[:-4] + ".txt")
        pred_list = api.txt_to_tuple_list(pred_file) if os.path.exists(pred_file) else []
        pred_list = [api.yolo_to_bbox(x, image.shape[1], image.shape[0]) for x in pred_list]

        gt_file = os.path.join(original_val_labels_dir, i[:-4] + ".txt")
        gt_list = api.txt_to_tuple_list(gt_file)
        gt_list = [x + (1,) for x in gt_list]
        gt_list = [api.yolo_to_bbox(x, image.shape[1], image.shape[0]) for x in gt_list]

        _, metrics = api.evaluate_detections(pred_list, gt_list, 0.25)
        f1_scores[idx] = metrics["f1_score"]

    nb_worst = max(original_val_size // 10, 3)
    nb_supervised = round(sample_size * supervised_ratio)
    nb_diverse = sample_size - nb_supervised
    worst_indices = np.argsort(f1_scores)[:nb_worst]
    # api.show_images(original_val_images_dir, original_val_images, worst_indices, "output")
    worst_avg_detections = [training_api.make_average_detection(os.path.join(original_val_images_dir, original_val_images[i]),
                                             os.path.join(original_val_labels_dir, original_val_labels[i]), seed=seed)
                      for i in worst_indices]
    worst_avg_detections = np.asarray(worst_avg_detections)
    worst_features = training_api.extract_features(worst_avg_detections)

    new_images_dir = os.path.join(dataset, "images")

    sys.argv = f"inference_pipeline.py --input_folder {new_images_dir} --model {model} " \
               f"--detection_only --silent --write_conf".split()
    args = inference_pipeline.parse_args()
    inference_pipeline.main(args)

    all_new_images = api.get_images(new_images_dir)
    new_images, new_labels = api.get_images_and_labels(new_images_dir, "output")
    max_to_consider = max(sample_size, max_to_consider)
    if len(new_images) > max_to_consider:
        indices_to_consider = uniform_partition(dataset, None, None, max_to_consider, seed)
        new_images = [new_images[i] for i in indices_to_consider]
        new_labels = [new_labels[i] for i in indices_to_consider]
    new_avg_detections = [training_api.make_average_detection(os.path.join(new_images_dir, new_images[i]),
                                                          os.path.join("output", new_labels[i]),
                                                          seed=seed)
                      for i in range(len(new_images))]
    new_avg_detections = np.asarray(new_avg_detections)
    new_features = training_api.extract_features(new_avg_detections)

    dists = pairwise_distances(new_features, worst_features)
    min_dists = dists.min(axis=1)
    supervised_indices = list(np.argsort(min_dists)[:nb_supervised])
    supervised_indices = api.convert_valid_to_real_indices(supervised_indices, new_images, all_new_images)
    indices = list(supervised_indices)
    # api.show_images(new_images_dir, new_images, supervised_indices)

    diverse_indices = diverse_sample(dataset, original_dataset, model, nb_diverse, seed)
    # api.show_images(new_images_dir, new_images, diverse_indices)
    n_random = 0
    for idx in diverse_indices:
        if idx not in supervised_indices:
            indices.append(idx)
        else:
            n_random += 1

    random.seed(seed)

    if len(new_images) - len(indices) < n_random:
        raise ValueError("Demanded sample size is greater than the dataset to sample from"
                         "(with empty boxes discarded). Please ask for a smaller sample size")
    remaining_indices = [all_new_images.index(new_images[i]) for i in range(len(new_images))
                         if all_new_images.index(new_images[i]) not in indices]
    random.shuffle(remaining_indices)
    for i in range(n_random):
        indices.append(remaining_indices[i])

    # api.show_images(new_images_dir, new_images, indices)

    return indices


def test_sample(sampling_method, model, sample_name="test"):
    print(f"Preparing selection {sample_name}...")
    original_dataset = "dataset0"
    remaining_dataset = "remaining"

    labels_dir = os.path.join(remaining_dataset, "labels")
    images_dir = os.path.join(remaining_dataset, "images")

    indices = sampling_method(remaining_dataset, original_dataset, model, 30, 69)
    api.make_set_from_indices(sample_name, images_dir, labels_dir, indices)

# test_sample(random_sample, None, "random_selection")
# test_sample(uniform_partition, None, "uniform_partition")
# test_sample(max_mean_uncertainty_sample, "dataset0.pt", "max_uncertainty")
# test_sample(diverse_sample, "dataset0.pt", "diverse")
# test_sample(supervised_sample, "dataset0.pt", "super")
