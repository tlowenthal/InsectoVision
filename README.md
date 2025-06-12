# InsectoVision
Welcome to the InsectoVision github repository !

As of June 2025, this project provides the state-of-the-art for insect detection in entomological boxes.
This work provides foundational tools for a complete and automatic processing of digital entomological collections.
Specifically, it allows accurate detection and localization of individual insects in entomological boxes, and does
so by carefully fine-tuning YOLOv8s on a tiny training set of 139 images (not provided in the project). Moreover, 
we pave the way for future dataset enlargement with the implementation of various Active Learning (AL) techniques.

Therefore, the project is ideal both for entomologists and for programmers who wish to optimally fine-tune YOLO 
on their very small single-class custom dataset : by using dynamic multi-phase fine-tuning, and by incrementally 
enlarging your labeled dataset using our AL sampling strategies, you will be able to reach good performance with 
very few training instances.

The project is completely open-source, and every step from pre-processing to training and inference is detailed and
commented.

## Tools overview
### Box Annotation UI

### Inference tool

Given a folder of entomological boxes' images, this script runs our detector's inference on each of them and
stores the YOLO format predictions in an 'output' folder. Several parameters can be chosen to choose the desired
inference pipeline.

### Training tool

Given a YOLO-format dataset (with an 'images' and a 'labels' subfolder), this script runs training using dynamic
multi-phase fine-tuning on YOLOv8s (or any other model initialization given as parameter). All training hyper-
parameters can be chosen freely using command-line arguments. Resulting models are stored into 'output.pt' and
'output.keras'.

### Memory analysis tool

Given your available hardware, and more specifically your GPU memory, this tool will perform grid-search on every
YOLO model size, and decide what is the optimal training image resolution for each of them. For example, 
the training configuration of our detector 'final_23.pt' was chosen using this script : we found that training 
YOLOv8s on 640*640 images would use 6.2GB. Given our 8GB RAM, this configuration optimally leveraged our
hardware. The results of the analysis are printed into STDOUT.

### AL selection tool

Given an unlabeled pool of images, a detection model and its labeled training set, this tool selects an optimal
sample of the unlabeled pool, to be merged into the training set for subsequent training. Selection is made
according to an AL strategy given as a command-line argument.

### AL re-training tool

Given a YOLO-format labeled dataset, the model trained on that dataset, and a new selection of labeled 
images (supposedly selected with AL selection), this tool merges the new set into the original training set and
performs fine-tuning of the model on that new enlarged dataset.

### Performance analysis tool

Given images, ground-truth labels, and model predictions, this tool computes the average precision, recall,
f1-score, map50 and map50-95 per image ,and prints the results into STDOUT.

## Use

Note that for more precise information, all scripts are runnable with the command-line argument '--help', which
will explain the usage of all arguments in more details.

### Box annotation UI
### Inference tool

The input images folder specified in '--input_folder' must contain images in jpg format. Typical inference 
can be run with the following command :
'python inference_pipeline.py --input_folder my_images'
Several arguments can be added for a more controllable inference pipeline, and we list some of them here :
    - write_conf : this argument is recommended if you want confidence level information into your txt
                   output label files. Corresponding confidence levels will be written at the end of each 
                   line. This information is essential for the UI as well as for the performance analysis tool.
    - model : the custom .pt detector you wish to use.
    - classifier : the custom .keras post-detection binary classifier you wish to use.
    - img_size : if you use a custom model on a resolution different from 640*640, you should specify image
                 size here.
    - detection_only : this skips the costly post-detection filtering made by the binary classifier, enabling
                       much faster inference with slightly reduced performances.
    - silent : if you do not want to be notified at directory deletions. Specifically, any already present 
               'output' folder will be completely replaced without any warning.

### Training tool

The input dataset in '--dataset' must be in standard YOLO format (with subfolders 'images' and 'labels').
Annotation files follow the YOLO format. Every line of the txt files corresponds to one bounding box,
of the form :
<class_id> <x_center> <y_center> <width> <height>
Typical training can be performed with the following command :
'python training_pipeline.py --dataset my_yolo_format_dataset'
Resulting detectors and post-detection binary classifiers will be stored into 'output.pt' and 'output.keras'.
Note that the former command will only work for macos. If running on windows or linux, specify '--gpu 0'.
The chosen value will be put into the '--device' argument of YOLO training. More information at this link :
https://docs.ultralytics.com/modes/train/
Several arguments can be added for a more controllable training pipeline, and we list some of them here :
    - fine_tuning_steps : number of steps for dynamic multi-phase fine-tuning. We advise a minimum of 3,
                          and it can be at maximum 23, because there are only 22 model layers to unfreeze.
    - lr0 : the initial learning rate at the first fine-tuning step (default: 0.01).
    - batch_init : initial batch size (default: 16).
    - batch_min : minimum batch size (default: 8). If you do not want batch size decreasing, choose
                  batch_min = batch_init
    - epochs : number of epochs at every fine-tuning step (default: 20)
    - patience : patience at every step (default: 5)
    - tp_ratio : the true positive ratio you wish in the training set of your binary post-detection classifier.
                 Default is 0.8, because it was found that training on a balanced set failed to converge.
    - model : weight initialization for detector's training (default: yolov8s.pt)
    - detection_only : skips the post-detection classifier's training.
    - classification_only : skips the detector's training, and only trains the classifier using the predictions
                            of the detector specified in '--model'.
    - replace_all : if you do not want to be notified at directory/file deletions. Specifically, any already 
                    present 'classify' folder and 'output.pt', 'output.keras' files will be completely replaced 
                    without any warning.

### Memory analysis tool





