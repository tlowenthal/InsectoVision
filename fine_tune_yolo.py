import argparse
import os
import shutil
import subprocess
import pandas as pd
import training_api

dataset = "yolo_compatible_dataset_cam/dataset.yaml"
run_dir = os.path.join("runs", "detect")
abs_run_path = os.path.join(os.getcwd(), run_dir)

def train_yolo(freeze_layers, batch_size, lr0, model_path, dataset, gpu, patience, epochs, img_size):
    path_to_data = os.path.join(dataset, "data.yaml")
    command = (
        f"yolo task=detect mode=train model={model_path} "
        f"data={path_to_data} project={abs_run_path} epochs={epochs} imgsz={img_size} "
        f"plots=True device={gpu} batch={batch_size} optimizer=Adam patience={patience} "
        f"freeze={freeze_layers} lr0={lr0} lrf={0.01}"
    )
    print(f"Running: {command}")
    subprocess.run(command, shell=True)


def log_best_map(run_id, results, batch_size, step, metric='metrics/mAP50(B)'):
    #results_path = os_independent_path(run_id, "results.csv", run_dir_offset)
    results_path = os.path.join(run_dir, "train" if run_id == 1 else f"train{run_id}", "results.csv")

    if os.path.exists(results_path):
        df = pd.read_csv(results_path)
        best_row = df.loc[df[metric].idxmax()]
        best_map50 = best_row[metric]
        tup_to_append = (run_id, 22 - (run_id - 1)*step, batch_size, best_map50)
        if metric == 'metrics/mAP50(B)':
            best_map5095 = best_row['metrics/mAP50-95(B)']
            tup_to_append += (best_map5095, )

        results.append(tup_to_append)
    else:
        print(f"Warning: {results_path} not found. Skipping log entry.")

# def os_independent_path(run_id, file_name, offset=2):
#     up_levels_path = ""
#     for i in range(offset):
#         up_levels_path = os.path.join(up_levels_path, "..")
#     train_folder = "train" if run_id == 1 else f"train{run_id}"
#     path = os.path.join(up_levels_path, "runs", "detect", train_folder, file_name)
#     return path


def main(args):
    # run_dir_offset = training_api.search_for_upwards_offset("runs")
    # if run_dir_offset != -1:
    #     up_levels_path = ""
    #     for i in range(run_dir_offset):
    #         up_levels_path = os.path.join(up_levels_path, "..")
    #     runs_path = os.path.join(up_levels_path, "runs")
    #     shutil.rmtree(runs_path)

    if os.path.exists(run_dir):
        shutil.rmtree(run_dir)
    os.makedirs(run_dir)

    # Initial settings
    freeze_layers = 22
    batch_size = args.batch_init
    lr0 = args.lr0
    results = []
    batch_size_list = [int(batch_size)]

    # First run with yolov8s.pt
    train_yolo(freeze_layers, batch_size, lr0, args.model, args.dataset, args.gpu, args.patience, args.epochs, args.img_size)

    step = 22 // (args.fine_tuning_steps - 1)
    lr_decrease_factor = (args.lr_final / lr0) ** (1 / args.fine_tuning_steps)

    # Run subsequent training iterations from 21 to 0
    for run_id, freeze_layers in enumerate(range(22 - step, -1, -step), start=2):
        # Gradually decrease learning rate
        lr0 *= lr_decrease_factor

        # Decrease batch size, but not below 8
        batch_size = max(args.batch_min, batch_size - args.batch_decrease_rate)
        batch_size_list.append(round(batch_size))

        # if run_dir_offset == -1:
        #     run_dir_offset = training_api.search_for_upwards_offset("runs")
        # model_path = os_independent_path(run_id - 1, os.path.join("weights", "best.pt"), run_dir_offset)
        model_path = os.path.join(run_dir, "train" if run_id == 2 else f"train{run_id - 1}", "weights", "best.pt")
        train_yolo(freeze_layers, round(batch_size), lr0, model_path, args.dataset,
                   args.gpu, args.patience, args.epochs, args.img_size)

    for rid in range(1, args.fine_tuning_steps + 1):
        log_best_map(rid, results, batch_size_list[rid - 1], step, 'metrics/recall(B)')

    # Print final results table
    print("\nFinal Training Results:")
    print("{:<5} {:<10} {:<8} {:<12}".format("Run", "Freeze", "Batch", "mAP50"))
    print("-" * 50)
    for run_id, freeze, batch, map50 in results:
        print(f"{run_id:<5} {freeze:<10} {batch:<8} {map50:<12.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="python fine_tune_yolo.py --dataset my_dataset")

    # Add command-line arguments
    parser.add_argument(
        "--fine_tuning_steps",
        type=int,
        default=5,
        help="Number of fine-tuning runs, with a constant rate of layer "
             "unfreezing down to 0 frozen layers (default: 5)"
    )
    parser.add_argument(
        "--gpu",
        type=str,
        default="mps",
        help="Gpu to use, the default is the macos standard (default: mps)"
    )
    # parser.add_argument(
    #     "--run_dir_offset",
    #     type=int,
    #     default=2,
    #     help="Depending on your machine and project architecture, the \'runs\' directory could be created"
    #          " in this script's directory (offset 0), or in its parent directory (offset 1), and so on. "
    #          "Typically, it will be in the grand-parent directory (default: 2)"
    # )
    parser.add_argument(
        "--lr0",
        type=float,
        default=0.01,
        help="Initial learning rate (default: 0.01)"
    )
    parser.add_argument(
        "--lr_final",
        type=float,
        default=-1,
        help="Final run learning rate (default: lr0/10)"
    )
    parser.add_argument(
        "--batch_init",
        type=int,
        default=16,
        help="Initial batch size (default: 16)"
    )
    parser.add_argument(
        "--batch_min",
        type=int,
        default=8,
        help="Minimal batch size from which it will start plateauing (default: 8)"
    )
    parser.add_argument(
        "--batch_decrease_rate",
        type=float,
        default=1,
        help="Rate at which batch size is reduced for each subsequent run (default: 1 per run)"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=20,
        help="Maximal number of epochs per run (default: 20)"
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=5,
        help="Patience fo early stopping (default: 5)"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="model/yolov8s.pt",
        help="Pretrained detection model to fine-tune on dataset (default: yolov8s.pt)"
    )
    parser.add_argument(
        "--img_size",
        type=int,
        default=640,
        help="Detector's input image size (default: 640)"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Path to the dataset in yaml format"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose output"
    )

    # Parse arguments and run the main function
    args = parser.parse_args()
    main(args)