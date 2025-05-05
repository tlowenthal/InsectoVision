import cv2
import tensorflow as tf
import numpy as np
import os
import api
import random
import training_api
from sklearn.metrics import pairwise_distances
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.applications.efficientnet import preprocess_input
from tensorflow.keras.models import Model

n_avg = 5

seed = 69
random.seed(seed)
np.random.seed(seed)

dataset = "whole_dataset"
labels_dir = os.path.join(dataset, "labels")
images_dir = os.path.join(dataset, "images")

images, labels = api.get_images_and_labels(images_dir, labels_dir)
images.sort()
labels.sort()

def make_average_detection(image_path, label_path, seed=seed):
    np.random.seed(seed)
    random.seed(seed)

    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    gt_file = label_path
    gt_list = api.txt_to_tuple_list(gt_file)
    gt_list = [x + (1,) for x in gt_list]
    gt_list = [api.yolo_to_bbox(x, image.shape[1], image.shape[0]) for x in gt_list]
    random.shuffle(gt_list)
    gt_list = gt_list[:n_avg] if len(gt_list) > n_avg else gt_list

    resized_pred_regions = []
    for region in gt_list:
        x_min, y_min, x_max, y_max, _ = map(int, region)
        cropped_region = image[y_min:y_max, x_min:x_max].astype(np.uint8)
        cropped_region = tf.image.resize(cropped_region, (256, 256), method=tf.image.ResizeMethod.BILINEAR)
        resized_pred_regions.append(cropped_region)
    resized_pred_regions = np.asarray(resized_pred_regions, dtype=np.uint8)
    return np.mean(resized_pred_regions, axis=0)


avg_detections = [make_average_detection(os.path.join(images_dir, images[i]),
                                       os.path.join(labels_dir, labels[i])) for i in range(len(images))]
avg_detections = np.asarray(avg_detections)

def select_diverse_images(images, n, seed=seed):
    """
    Select n most diverse images from a numpy array using greedy max-min pixel-level distance.

    Parameters:
        images (np.ndarray): Array of shape (N, H, W[, C])
        n (int): Number of diverse images to select

    Returns:
        diverse_images (np.ndarray): Subset of selected images
        selected_indices (list): Indices of selected images
    """
    np.random.seed(seed)
    random.seed(seed)

    # Flatten images to vectors
    N = len(images)
    flat_images = images.reshape(N, -1)

    # Start with a random image
    selected_indices = [np.random.randint(N)]
    selected_features = flat_images[selected_indices]

    for _ in range(n - 1):
        remaining_indices = list(set(range(N)) - set(selected_indices))
        remaining_features = flat_images[remaining_indices]

        # Compute distances to the selected set
        dists = pairwise_distances(remaining_features, selected_features)
        min_dists = dists.min(axis=1)

        # Pick the image with the largest minimum distance
        next_idx = remaining_indices[np.argmax(min_dists)]
        selected_indices.append(next_idx)
        selected_features = flat_images[selected_indices]

    diverse_images = images[selected_indices]
    return diverse_images, selected_indices

n = 30
step = 5
#_, indices = select_diverse_images(avg_detections, n)
#api.make_test_set("test_pixelwise", images_dir, labels_dir, indices)
# titles = [images[i] for i in indices]
# for i in range(0, n, step):
#     training_api.plot_images(diverse_images[i:i+step], titles[i:i+step])


def extract_features(images, batch_size=32):
    """
    Extract features from images using EfficientNetB0 in TensorFlow.

    Args:
        images (np.ndarray): (N, H, W, C), dtype uint8 or float32
        batch_size (int): Batch size for inference

    Returns:
        features (np.ndarray): (N, D) feature vectors
    """
    base_model = EfficientNetB0(weights='imagenet', include_top=False, pooling='avg')
    model = Model(inputs=base_model.input, outputs=base_model.output)

    # Ensure input is float32 and scaled to [0, 255]
    if images.dtype != np.float32:
        images = images.astype(np.float32)

    features = []
    num_batches = int(np.ceil(len(images) / batch_size))

    for i in range(num_batches):
        batch = images[i * batch_size : (i + 1) * batch_size]
        # Resize to 224x224 and preprocess
        batch_resized = tf.image.resize(batch, (224, 224))
        batch_preprocessed = preprocess_input(batch_resized)
        batch_features = model(batch_preprocessed, training=False).numpy()
        features.append(batch_features)

    return np.concatenate(features, axis=0)


features = extract_features(avg_detections)
_, indices = select_diverse_images(features, n)
print(indices)
api.make_set_from_indices("final_test_set", images_dir, labels_dir, indices)
api.make_set_from_indices("final_training_set", images_dir, labels_dir,
                          [i for i in range(len(labels)) if i not in indices])
# diverse_images = [avg_detections[i] for i in indices]
# titles = [images[i] for i in indices]
# for i in range(0, n, step):
#     training_api.plot_images(diverse_images[i:i+step], titles[i:i+step])


def make_representative_split(images, labels, split_ratio, seed=seed):
    images = list(images)
    labels = list(labels)
    images.sort()
    labels.sort()

    avg_detections = [make_average_detection(os.path.join(images_dir, images[i]),
                                             os.path.join(labels_dir, labels[i]), seed=seed)
                      for i in range(len(images))]
    avg_detections = np.asarray(avg_detections)

    n = int(len(images) * split_ratio)

    features = extract_features(avg_detections)
    _, indices = select_diverse_images(features, n, seed=seed)
    test_images = [images[i] for i in indices]
    test_labels = [labels[i] for i in indices]
    train_images = [images[i] for i in range(len(images)) if i not in indices]
    train_labels = [labels[i] for i in range(len(labels)) if i not in indices]

    return train_images, train_labels, test_images, test_labels

