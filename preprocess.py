import numpy as np
import tensorflow as tf

def preprocess_max_dim(images, target_size=(434, 311)):
    ret = []
    for image in images:
        # Get the current image shape
        height, width = image.shape[:2]

        # Calculate the padding for width and height
        pad_width = (target_size[1] - width) // 2
        pad_height = (target_size[0] - height) // 2

        # Create a new image filled with white pixels
        padded_image = np.ones((target_size[0], target_size[1], 3), dtype=np.float32) * 255

        # Place the original image in the center of the padded image
        padded_image[pad_height:pad_height + height, pad_width:pad_width + width] = image

        ret.append(padded_image)
    return np.asarray(ret)

def preprocess_512(images, target_size=(512, 512)):
    ret = []
    for image in images:
        image = tf.image.resize_with_pad(image, target_size[0], target_size[1])
        ret.append(image)
    return np.asarray(ret, dtype=np.float32)

def preprocess_256(images):
    return preprocess_512(images, (256, 256))

dic = {"vanillaPredictor.keras" : preprocess_512,
       "fcnn.keras" : preprocess_max_dim,
       "fcnn_good_dim.keras" : preprocess_256,
       "vanilla_fcn.keras" : preprocess_256,
       "pretrained.keras" : preprocess_256}