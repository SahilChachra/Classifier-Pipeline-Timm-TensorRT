# Template for Multiclass classifier in PyTorch for Classification with Timm model Implementation and TensorRT Export

## Features/Options/Support
1. Custom model file with Timm models
2. Custom dataset file
3. Creates Experiment folder allowing you to run continous training jobs.
4. Support training for dataset stored in folder or in CSV
5. Categorical Labels - [1], [2], [3]...
7. Displays Training condiguration so you cross check the input
8. Run validation on Train and test set and saves ConfusionMatrix as PNG.
10. Use of garbage collector and torch's method to clear GPU cache
5. Early stopping to save your time and resources
6. Saves model checkpoint and weight file as accuracy improves
9. Albumentations for image augmentation
11. Added options for Loss functions
3. Multiple LR Schedulers
0. Added WandB support
0. Added Number of workers parameter
0. Added Seed to help reproduce experiment
0. Added Torch inference code
0. Added Torch to onnx model export
0. Added Onnx inference code
0. Added Onnx to TensorRT conversion code
0. Added TensorRT inference code

## To/Do

1. Heatmap of features
2. Add LabelSmoothing
3. Add Gradient Clipping
4. Add Mixed Precision Training

## References
1. [Scene Classification Dataset](https://www.kaggle.com/datasets/nitishabharathi/scene-classification)
2. Kaggle Notebook : [Transfer Learning with Timm](https://www.kaggle.com/code/hinepo/transfer-learning-with-timm-models-and-pytorch)
3. Kaggle Notebook : [EfficientNet Mixup Leak free](https://www.kaggle.com/code/debarshichanda/efficientnetv2-mixup-leak-free)
4. Kaggle Notebook : [Scene classification](https://www.kaggle.com/code/krishnayogi/scene-classification-using-transfer-learning)
5. Convert PyTorch model to TensorRT - [Link](https://learnopencv.com/how-to-convert-a-model-from-pytorch-to-tensorrt-and-speed-up-inference/)
6. Getting started with PyTorch model & Timm - [Link](https://towardsdatascience.com/getting-started-with-pytorch-image-models-timm-a-practitioners-guide-4e77b4bf9055#9388)

## :heart: Made by [Sahil Chachra](https://github.com/SahilChachra)