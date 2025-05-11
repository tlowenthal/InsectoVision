import os
import random

import cv2
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
from tensorflow import keras as tfk
from tensorflow.keras import layers as tfkl
import tensorflow as tf
from sklearn.metrics import pairwise_distances
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.applications.efficientnet import preprocess_input
from tensorflow.keras.models import Model

import api


def save_tps_and_fps(images, predictions, ground_truth, output_dir, resize_mode='bilinear'):
    output_tp_length = 0
    output_fp_length = 0
    output_tp_folder = os.path.join(output_dir, "tps")
    output_fp_folder = os.path.join(output_dir, "fps")
    os.makedirs(output_tp_folder)
    os.makedirs(output_fp_folder)
    for idx, i in enumerate(os.listdir(images)):

        image = cv2.imread(os.path.join(images, i))

        pred_list = api.txt_to_tuple_list(os.path.join(predictions, i[:-4] + ".txt"))
        if len(pred_list) > 0 and len(pred_list[0]) == 4:
            pred_list = [x + (1,) for x in pred_list]
        pred_list = [api.yolo_to_bbox(x, image.shape[1], image.shape[0]) for x in pred_list]

        gt_list = api.txt_to_tuple_list(os.path.join(ground_truth, i[:-4] + ".txt"))
        gt_list = [x + (1,) for x in gt_list]
        gt_list = [api.yolo_to_bbox(x, image.shape[1], image.shape[0]) for x in gt_list]

        fp_list, metrics = api.evaluate_detections(pred_list, gt_list, 0.25)
        tp_list = [x for x in pred_list if x not in fp_list]

        output_tp_length = api.save_regions(image, tp_list, output_tp_folder, output_tp_length,
                                            resize=(256, 256), resize_mode=resize_mode)
        output_fp_length = api.save_regions(image, fp_list, output_fp_folder, output_fp_length,
                                            resize=(256, 256), resize_mode=resize_mode)


def make_image_and_label_array(dir):
    # Path to image folder
    image_folder_class_0 = os.path.join(dir, "fps")
    image_folder_class_1 = os.path.join(dir, "tps")

    # Load images into a list
    x = []
    y = []

    for i, image_folder in enumerate((image_folder_class_0, image_folder_class_1)):

        image_filenames = os.listdir(image_folder)  # Sorting ensures consistency
        image_filenames.sort(key=lambda f: int(f[1:-4]))

        for filename in image_filenames:
            if filename.endswith((".png", ".jpg", ".jpeg", ".JPG")):  # Check for valid image formats
                img_path = os.path.join(image_folder, filename)
                img = Image.open(img_path)  # Open image
                img_array = np.array(img, dtype=np.float32)  # Convert to numpy array
                x.append(img_array)
                y.append(i)

    x = np.asarray(x)
    y = np.asarray(y, dtype=np.float32)

    return x, y

def build_convnet(input_shape=(256,256,3), learning_rate=0.001):
    convnet = tfk.applications.EfficientNetB0(
        input_shape=(256, 256, 3),
        include_top=False,
        weights="imagenet"
    )

    # Use the supernet as feature extractor, i.e. freeze all its weigths
    convnet.trainable = False

    # Create an input layer with shape (224, 224, 3)
    inputs = tfk.Input(shape=input_shape)

    # Connect ConvNeXt to the input
    x = convnet(inputs, training = False)
    #dropout = tfkl.Dropout(0.1)(x)  # Regularize with dropout

    x = tfkl.GlobalAveragePooling2D()(x)  # Reduce dimensionality
    x = tfkl.BatchNormalization()(x)

    dense1 = tfkl.Dense(units=256, activation='gelu',name='dense1')(x)
    dense1_dropout = tfkl.Dropout(0.2, name = 'dense1_dropout')(dense1)  # Regularize with dropout

    dense2 = tfkl.Dense(units=128, activation='gelu',name='dense2')(dense1_dropout)
    dense2_dropout = tfkl.Dropout(0.2, name = 'dense2_dropout')(dense2)  # Regularize with dropout

    # Add a Dense layer with 2 units and softmax activation as the classifier
    outputs = tfkl.Dense(2, activation='softmax', name = 'output')(dense2_dropout)

    # Create a Model connecting input and output
    model = tfk.Model(inputs=inputs, outputs=outputs, name='conv_model')

    # Compile the model with Binary Cross-Entropy loss and AdamW optimizer
    optimizer = tfk.optimizers.Adam(learning_rate=learning_rate)
    model.compile(loss=tfk.losses.BinaryCrossentropy(), optimizer=optimizer, metrics=['accuracy'])

    return model

def plot_images(images, titles=None, max_cols=5, figsize=(12, 6)):
    """
    Plots multiple images in a grid layout.

    Args:
    - images (list or array): List of images (H, W, C) in range [0,255] or [0,1].
    - titles (list): Optional list of titles for each image.
    - max_cols (int): Max columns in the grid (default=5).
    - figsize (tuple): Figure size.
    """
    num_images = len(images)
    cols = min(num_images, max_cols)
    rows = (num_images // cols) + (num_images % cols > 0)  # Compute rows dynamically

    fig, axes = plt.subplots(rows, cols, figsize=figsize)

    # Ensure axes is always a 2D array for easy indexing
    axes = np.array(axes).reshape(rows, cols)

    for i, ax in enumerate(axes.flat):
        if i < num_images:
            img = images[i].astype(np.uint8)  # Ensure correct dtype
            ax.imshow(img, vmin=0, vmax=255)
            if titles:
                ax.set_title(titles[i], fontsize=10)
        ax.axis("off")  # Hide axes

    plt.tight_layout()
    plt.show()

def extract_heatmap(img_array, heatmap, alpha=0.1):

    heatmap  = tf.image.resize(heatmap, (img_array.shape[0], img_array.shape[1]), method=tf.image.ResizeMethod.BILINEAR)
    heatmap = np.uint8(heatmap * 255)  # Scale to [0,255]
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)  # Apply color map
    heatmap_RGB = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    superimposed_img_BGR = cv2.addWeighted(np.uint8(img_array), 1 - alpha, heatmap_RGB, alpha, 0)  # Blend images
    superimposed_img = cv2.cvtColor(superimposed_img_BGR, cv2.COLOR_BGR2RGB)

    return superimposed_img


seed=69
def make_average_detection(image_path, label_path, seed=seed, n_avg=5):
    np.random.seed(seed)
    random.seed(seed)

    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    gt_file = label_path
    gt_list = api.txt_to_tuple_list(gt_file)
    if len(gt_list) > 0 and len(gt_list[0]) == 4:
        gt_list = [x + (1,) for x in gt_list]
    gt_list = [api.yolo_to_bbox(x, image.shape[1], image.shape[0]) for x in gt_list]
    random.shuffle(gt_list)
    gt_list = sorted(gt_list, key=lambda x: x[4], reverse=True)
    gt_list = gt_list[:n_avg] if len(gt_list) > n_avg else gt_list

    resized_pred_regions = []
    for region in gt_list:
        x_min, y_min, x_max, y_max, _ = map(int, region)
        cropped_region = image[y_min:y_max, x_min:x_max].astype(np.uint8)
        cropped_region = tf.image.resize(cropped_region, (256, 256), method=tf.image.ResizeMethod.BILINEAR)
        resized_pred_regions.append(cropped_region)
    if len(resized_pred_regions) == 0:
        return np.zeros((256, 256, 3), dtype=np.uint8) # black image
    resized_pred_regions = np.asarray(resized_pred_regions, dtype=np.uint8)
    return np.mean(resized_pred_regions, axis=0)


def select_diverse_images(images, n, seed=seed, original_features=None):
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
        to_diverge_from = np.append(original_features, selected_features, axis=0) \
            if original_features is not None else selected_features
        dists = pairwise_distances(remaining_features, to_diverge_from)
        min_dists = dists.min(axis=1)

        # Pick the image with the largest minimum distance
        next_idx = remaining_indices[np.argmax(min_dists)]
        selected_indices.append(next_idx)
        selected_features = flat_images[selected_indices]

    diverse_images = images[selected_indices]
    return diverse_images, selected_indices


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


def make_representative_split(images_dir, labels_dir, test_size, seed=seed, original_features=None):

    images, labels = api.get_images_and_labels(images_dir, labels_dir)

    avg_detections = [make_average_detection(os.path.join(images_dir, images[i]),
                                             os.path.join(labels_dir, labels[i]), seed=seed)
                      for i in range(len(images))]
    avg_detections = np.asarray(avg_detections)

    features = extract_features(avg_detections)
    _, indices = select_diverse_images(features, test_size, seed=seed, original_features=original_features)
    return indices
