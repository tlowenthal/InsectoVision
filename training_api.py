import os
import cv2
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
from tensorflow import keras as tfk
from tensorflow.keras import layers as tfkl

import api


def save_tps_and_fps(images, predictions, ground_truth, output_dir):
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

        output_tp_length = api.save_regions(image, tp_list, output_tp_folder, output_tp_length, resize=(256, 256))
        output_fp_length = api.save_regions(image, fp_list, output_fp_folder, output_fp_length, resize=(256, 256))


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

def build_convnet(input_shape=(256,256,3)):
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
    model.compile(loss=tfk.losses.BinaryCrossentropy(), optimizer="adam", metrics=['accuracy'])

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