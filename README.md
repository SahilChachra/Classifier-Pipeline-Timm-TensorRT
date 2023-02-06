# Pipeline for training Multiclass classifier in PyTorch with Timm model library and TensorRT/Onnx Export

## :question: How to use
The script expects images/labels info stored in CSV file and dataset in folder. Check Scene Classification dataset example. Link in reference

Command : `python3 train.py --model_name resnet50 --epochs 100 --batch_size 64 --lr 0.0001 --img_size 256 --device 0 --optimizer adam --lr_scheduler CosineAnnealingLR --dataset /home/SceneData --split 0.2 --target_size 3 --early_stop 10 --loss_func CrossEntropyLoss --save_checkpoint_folder ./checkpoints --save_model_folder ./weights --exp_name testExp --labels night,day,noon --wandb --projec_name SceneClassifier --seed 22 --workers 4`

If you want to train on dataset which is stored in folders then pass :
`python3 train.py OTHER_ARGUEMENTS_AS_MENTIONED_ABOVE --image_folder flag along with --train_image_folder trainFolder --val_image_folder validFolder`

In this case, dataset structure should be like -> \
DatasetName \
&emsp;|-> train_data \
&emsp;&emsp;|-> class_1 \
&emsp;&emsp;|-> class_2 \
&emsp;&emsp;|-> class_3 \
&emsp;|-> val_data \
&emsp;&emsp;|-> class_1 \
&emsp;&emsp;|-> class_2 \
&emsp;&emsp;|-> class_3 

## :fire: Features/Options/Support
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

## :golf: To/Do

1. Heatmap of features
2. Add LabelSmoothing
3. Add Gradient Clipping
4. Add Mixed Precision Training

## :anger: Set up your enviroment
1. Install venv (recommended)
2. Install all the requirements using requirements.txt
3. To install TensorRT, refer [Nvidia's Link](https://docs.nvidia.com/deeplearning/tensorrt/install-guide/index.html)

## :diamonds: References
1. [Scene Classification Dataset](https://www.kaggle.com/datasets/nitishabharathi/scene-classification)
2. Kaggle Notebook : [Transfer Learning with Timm](https://www.kaggle.com/code/hinepo/transfer-learning-with-timm-models-and-pytorch)
3. Kaggle Notebook : [EfficientNet Mixup Leak free](https://www.kaggle.com/code/debarshichanda/efficientnetv2-mixup-leak-free)
4. Kaggle Notebook : [Scene classification](https://www.kaggle.com/code/krishnayogi/scene-classification-using-transfer-learning)
5. Convert PyTorch model to TensorRT - [Link](https://learnopencv.com/how-to-convert-a-model-from-pytorch-to-tensorrt-and-speed-up-inference/)
6. Getting started with PyTorch model & Timm - [Link](https://towardsdatascience.com/getting-started-with-pytorch-image-models-timm-a-practitioners-guide-4e77b4bf9055#9388)

## :heart: Made by [Sahil Chachra](https://github.com/SahilChachra)