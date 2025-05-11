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
silent = True
iterations = 4
ft_steps = 10

def train(dataset, no_high_precision=True, no_split=True):
    sys.argv = f"training_pipeline.py --dataset {dataset} " \
               f"--verbose --replace_all --fine_tuning_steps {ft_steps}".split()
    if no_split:
        sys.argv.append("--no_split")
    training_args = training_pipeline.parse_args()
    training_pipeline.main(training_args)

    api.warn_user_if_file_exists(f"{dataset}.pt", silent=silent)
    api.warn_user_if_file_exists(f"{dataset}.keras", silent=silent)
    os.rename("output.pt", f"{dataset}.pt")
    os.rename("output.keras", f"{dataset}.keras")

    if not no_high_precision:
        sys.argv = f"training_pipeline.py --dataset {dataset} --heatmap_extractor {dataset}.keras " \
                   f"--verbose --replace_all --fine_tuning_steps {ft_steps}".split()
        training_args = training_pipeline.parse_args()
        training_pipeline.main(training_args)

        api.warn_user_if_file_exists(f"{dataset}_high_precision.pt", silent=silent)
        os.rename("output.pt", f"{dataset}_high_precision.pt")


def print_performance(images, labels, predictions, min_conf=0.0, no_map=False):
    sys.argv = f"performance.py --images {images} " \
               f"--ground_truth {labels} --predictions {predictions} --min_conf {min_conf}".split()
    if no_map:
        sys.argv.append("--no_map")
    args = performance.parse_args()
    performance.main(args)


def assess_performance(images_dir, labels_dir, model, classifier, high_precision_model=None, include_corrector=True):
    print(f"Inferring with detector {model}...\n")
    sys.argv = f"inference_pipeline.py --input_folder {images_dir} " \
               f"--silent --model {model} --detection_only --write_conf".split()
    args = inference_pipeline.parse_args()
    inference_pipeline.main(args)

    print(f"Performance of detector {model}:")
    print_performance(images_dir, labels_dir, "output")
    print()

    if include_corrector:

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
        sys.argv = f"inference_pipeline.py --input_folder {images_dir} --high_precision " \
                   f"--silent --model {high_precision_model} --classifier {classifier} --write_conf".split()
        args = inference_pipeline.parse_args()
        inference_pipeline.main(args)

        print(f"Performance of high-precision detector {high_precision_model}:")
        print_performance(images_dir, labels_dir, "output")
        print()



def fine_tune(model, original_dataset, new_images, no_high_precision=True):
    print(f"\n\nFine-tuning model {model}, which was trained with dataset {original_dataset}, "
          f"on new images {new_images}...\n\n")
    sys.argv = f"retrain.py --dataset {original_dataset} --new_images {new_images} " \
               f"--verbose --model {model} --original_nb_steps {ft_steps}".split()
    if no_high_precision:
        sys.argv.append("--no_high_precision")
    if silent:
        sys.argv.append("--silent")
    training_args = retrain.parse_args()
    retrain.main(training_args)

    api.warn_user_if_file_exists(f"{new_images}.pt", silent=silent)
    api.warn_user_if_file_exists(f"{new_images}.keras", silent=silent)
    os.rename("new_model.pt", f"{new_images}.pt")
    os.rename("new_classifier.keras", f"{new_images}.keras")
    if not no_high_precision:
        api.warn_user_if_file_exists(f"{new_images}_high_precision.pt", silent=silent)
        os.rename("new_high_precision_model.pt", f"{new_images}_high_precision.pt")
    shutil.rmtree(new_images)
    os.rename("new_dataset", new_images)


def merge_and_train(_, original_dataset, new_images, no_high_precision=True):
    print(f"\n\nMerging {original_dataset} with {new_images} and training normally...\n\n")
    api.merge_datasets(original_dataset, new_images, "new_dataset", seed=seed)
    shutil.rmtree(new_images)
    os.rename("new_dataset", new_images)
    train(new_images, no_high_precision=no_high_precision)


def simulate_active_learning(sampling_strategy=subsample.random_sample,
                             initial_sampling_strategy=subsample.uniform_partition, training_strategy=fine_tune,
                             high_precision=False, train_dataset0=True, log_directory=None):
    if log_directory is not None:
        api.warn_user_if_directory_exists(log_directory, silent=silent)

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
    if train_dataset0:
        api.make_set_from_indices("dataset0", images_dir, labels_dir, indices, silent=silent)
    api.make_set_from_indices("remaining", images_dir, labels_dir,
                              [i for i in range(len(images)) if i not in indices], silent=silent)

    if train_dataset0:
        train("dataset0", no_high_precision=not high_precision, no_split=False)
        api.warn_user_if_directory_exists("runs0", silent=silent, make_dir=False)
        shutil.move("runs", "runs0")

    for i in range(1, iterations):
        api.warn_user_if_directory_exists("current_remaining", silent=silent, make_dir=False)
        os.rename("remaining", "current_remaining")
        labels_dir = os.path.join(f"current_remaining", "labels")
        images_dir = os.path.join(f"current_remaining", "images")

        images, labels = api.get_images_and_labels(images_dir, labels_dir)

        indices = sampling_strategy("current_remaining", f"dataset{i - 1}",
                                    f"dataset{i - 1}.pt", sample_size[i], seed)
        api.make_set_from_indices(f"dataset{i}", images_dir, labels_dir, indices, silent=silent)
        api.make_set_from_indices("remaining", images_dir, labels_dir,
                                  [i for i in range(len(images)) if i not in indices], silent=silent)

        training_strategy(f"dataset{i - 1}.pt", f"dataset{i - 1}",
                          f"dataset{i}", no_high_precision=not high_precision)
        if log_directory is not None:
            api.warn_user_if_directory_exists(os.path.join(log_directory, f"runs{i}"), silent=silent, make_dir=False)
            shutil.move("runs", os.path.join(log_directory, f"runs{i}"))

        shutil.rmtree("current_remaining")


images_dir = os.path.join(dataset, "images")
labels_dir = os.path.join(dataset, "labels")
test_images_dir = os.path.join(test_set, "images")
test_labels_dir = os.path.join(test_set, "labels")
strategies = [subsample.random_sample, subsample.max_mean_uncertainty_sample, subsample.diverse_sample, subsample.supervised_sample]
strategy_names = ["random_sample", "uncertainty", "diversity", "supervised"]

#simulate_active_learning(train_dataset0=False)

for strat_id, strat in list(enumerate(strategies)):
    simulate_active_learning(sampling_strategy=strat, training_strategy=fine_tune, train_dataset0=False,
                             log_directory=strategy_names[strat_id])

    for i in range(iterations):
        prefix = f"dataset{i}"
        model = prefix + ".pt"
        classifier = prefix + ".keras"
        if i > 0:
            shutil.move(model, os.path.join(strategy_names[strat_id], model))
            shutil.move(classifier, os.path.join(strategy_names[strat_id], classifier))

print(f"\n### Evaluating performance of initial model trained on dataset0 ###\n".upper())
assess_performance(test_images_dir, test_labels_dir, "dataset0.pt", "dataset0.keras")
for strat in strategy_names:
    print(f"\n### Evaluating performance of sampling strategy {strat} ###\n".upper())
    for i in range(1, iterations):
        prefix = f"dataset{i}"
        model = os.path.join(strat, prefix + ".pt")
        classifier = os.path.join(strat, prefix + ".keras")
        assess_performance(test_images_dir, test_labels_dir, model, classifier, include_corrector=False)
        print()


#assess_performance(images_dir, labels_dir, "final_23.pt", "final_23.keras")