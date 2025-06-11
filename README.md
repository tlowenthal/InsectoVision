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