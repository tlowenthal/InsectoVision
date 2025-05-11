import os
import sys

import training_pipeline

fine_tuning_steps = 23
batch_init = 16
tp_ratio = 0.75
training_set = "whole_dataset"
output_name = "final"

# sys.argv = f"training_pipeline.py --batch_init {batch_init} --dataset {training_set} " \
#            f"--verbose --replace_all --fine_tuning_steps {fine_tuning_steps} --tp_ratio {tp_ratio}".split()
sys.argv = f"training_pipeline.py --model {output_name}_{fine_tuning_steps}.pt --dataset {training_set} " \
           f"--verbose --replace_all --classification_only --tp_ratio {tp_ratio}".split()
training_args = training_pipeline.parse_args()
training_pipeline.main(training_args)

#os.rename("output.pt", f"{output_name}_{fine_tuning_steps}.pt")
os.rename("output.keras", f"{output_name}_{fine_tuning_steps}.keras")

sys.argv = f"training_pipeline.py --dataset {training_set} --heatmap_extractor {output_name}_{fine_tuning_steps}.keras " \
           f"--verbose --replace_all --fine_tuning_steps {fine_tuning_steps} --batch_init {batch_init}".split()
training_args = training_pipeline.parse_args()
training_pipeline.main(training_args)

os.rename("output.pt", f"high_precision_{output_name}_{fine_tuning_steps}.pt")
