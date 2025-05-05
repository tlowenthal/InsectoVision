import os
import shutil

import numpy as np
import api
import random
import sys
import training_pipeline
import retrain
import inference_pipeline
import performance
import subsample

dataset = "whole_dataset"
test_set = "Test_Set"
seed = 69
random.seed(seed)
np.random.seed(seed)
silent = False
iterations = 4
ft_steps = 23

def train(dataset):
    sys.argv = f"training_pipeline.py --dataset {dataset} " \
               f"--verbose --replace_all --fine_tuning_steps {ft_steps}".split()
    training_args = training_pipeline.parse_args()
    training_pipeline.main(training_args)

    os.rename("output.pt", f"{dataset}.pt")
    os.rename("output.keras", f"{dataset}.keras")


def infer(images, model, classifier, output="output"):
    sys.argv = f"inference_pipeline.py --input_folder {images} " \
               f"--silent --model {model} --classifier {classifier}".split()
    args = inference_pipeline.parse_args()
    inference_pipeline.main(args)

    if output != "output":
        os.rename("output", output)


def print_performance(images, labels, predictions, min_conf=0.0, no_map=False):
    sys.argv = f"performance.py --images {images} " \
               f"--ground_truth {labels} --predictions {predictions} --min_conf {min_conf}".split()
    if no_map:
        sys.argv.append("--no_map")
    args = performance.parse_args()
    performance.main(args)


def assess_performance(images_dir, labels_dir, model, classifier, high_precision_model=None, min_conf=0.75):
    print(f"Inferring with detector {model}...\n")
    sys.argv = f"inference_pipeline.py --input_folder {images_dir} " \
               f"--silent --model {model} --detection_only --write_conf".split()
    args = inference_pipeline.parse_args()
    inference_pipeline.main(args)

    print(f"Performance of detector {model}:")
    print_performance(images_dir, labels_dir, "output")
    print()

    print(f"Performance of detector {model} with min_conf = {min_conf}:")
    print_performance(images_dir, labels_dir, "output", min_conf=min_conf, no_map=True)
    print()

    print(f"Inferring with detector {model} and corrector {classifier}...\n")
    sys.argv = f"inference_pipeline.py --input_folder {images_dir} " \
               f"--silent --model {model} --classifier {classifier} --write_conf".split()
    args = inference_pipeline.parse_args()
    inference_pipeline.main(args)

    print(f"Performance of detector {model} + corrector {classifier}:")
    print_performance(images_dir, labels_dir, "output", no_map=True)
    print()

    if high_precision_model is not None:
        print(f"Inferring with high_precision detector {high_precision_model}...\n")
        sys.argv = f"inference_pipeline.py --input_folder {images_dir} " \
                   f"--silent --model {high_precision_model} --heatmap_extractor {classifier} --write_conf".split()
        args = inference_pipeline.parse_args()
        inference_pipeline.main(args)

        print(f"Performance of high-precision detector {high_precision_model}:")
        print_performance(images_dir, labels_dir, "output")
        print()



def fine_tune(model, original_dataset, new_images):
    print(f"\n\nFine-tuning model {model}, which was trained with dataset {original_dataset}, "
          f"on new images {new_images}...\n\n")
    sys.argv = f"retrain.py --dataset {original_dataset} --new_images {new_images} " \
               f"--verbose --detection_only --model {model} --original_nb_steps {ft_steps}".split()
    training_args = retrain.parse_args()
    retrain.main(training_args)

    os.rename("output.pt", f"{new_images}.pt")
    os.rename("output.keras", f"{new_images}.keras")
    shutil.rmtree(new_images)
    os.rename("new_dataset", new_images)


def merge_and_train(_, original_dataset, new_images):
    print(f"\n\nMerging {original_dataset} with {new_images} and training normally...\n\n")
    retrain.merge_datasets(original_dataset, new_images, "new_dataset", seed=seed)
    shutil.rmtree(new_images)
    os.rename("new_dataset", new_images)
    train(new_images)


def evaluate_performance(sampling_strategy=subsample.random_sample,
                         initial_sampling_strategy=subsample.random_sample, training_strategy=fine_tune):
    labels_dir = os.path.join(dataset, "labels")
    images_dir = os.path.join(dataset, "images")

    images, labels = api.get_images_and_labels(images_dir, labels_dir)
    images.sort()
    labels.sort()

    sample_size = []
    for i in range(iterations - 1):
        sample_size.append(len(images) // iterations)
    sample_size.append(len(images) - (iterations - 1) * (len(images) // iterations))

    indices = initial_sampling_strategy(dataset, None, None, sample_size[0], seed)
    api.make_set_from_indices("dataset0", images_dir, labels_dir, indices, silent=silent)
    api.make_set_from_indices("remaining", images_dir, labels_dir,
                              [i for i in range(len(images)) if i not in indices], silent=silent)

    train("dataset0")

    for i in range(1, iterations):
        if not os.path.exists("current_remaining"):
            os.rename("remaining", "current_remaining")
        labels_dir = os.path.join(f"current_remaining", "labels")
        images_dir = os.path.join(f"current_remaining", "images")

        images, labels = api.get_images_and_labels(images_dir, labels_dir)
        images.sort()
        labels.sort()

        indices = sampling_strategy("current_remaining", f"dataset{i - 1}",
                                    f"dataset{i - 1}.pt", sample_size[i], seed)
        api.make_set_from_indices(f"dataset{i}", images_dir, labels_dir, indices, silent=silent)
        api.make_set_from_indices("remaining", images_dir, labels_dir,
                                  [i for i in range(len(images)) if i not in indices], silent=silent)

        training_strategy(f"dataset{i - 1}.pt", f"dataset{i - 1}", f"dataset{i}")

        shutil.rmtree("current_remaining")


images_dir = os.path.join(test_set, "images")
labels_dir = os.path.join(test_set, "labels")
# evaluate_performance()
# for i in range(iterations):
#     prefix = f"dataset{i}"
#     model = prefix + ".pt"
#     classifier = prefix + ".keras"
#     assess_performance(images_dir, labels_dir, model, classifier)
#     print()

assess_performance(images_dir, labels_dir, "final_3.pt", "final_3.keras")