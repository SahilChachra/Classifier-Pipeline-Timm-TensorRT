from configparser import Interpolation
import numpy as np
import os
import cv2
import onnx
import onnxruntime as ort
import argparse
from loguru import logger

@logger.catch()
def main(args):
    model_path = args.model_path
    image_path = args.image_path
    labels = args.labels
    batch_size = args.batch_size
    img_size = args.img_size
    target_size = args.target_size

    # Some checks
    if ort.get_device() == "GPU":
        logger.info("Inferencing on GPU")
        device = "GPU"
    else:
        logger.info("Inferencing on GPU")
        device = "CPU"

    providers = ["CUDAExecutionProvider"] if device=="GPU" else ["CPUExecutionProvider"]

    # Set up labels
    label_temp = labels.split(",")
    labels = [i.strip() for i in label_temp]

    # Load model
    try:
        model = onnx.load(model_path)
        logger.info("Checking model...")
        onnx.checker.check_model(model)
        onnx.helper.printable_graph(model.graph)
        logger.info("Model checked...")
        
        logger.info("Running inference...")
        
        ort_session = ort.InferenceSession(model_path, providers=providers)

        img_list = []
        for image in os.listdir(image_path):
            img = cv2.imread(os.path.join(image_path, image), cv2.IMREAD_COLOR)
            img = cv2.resize(img, (img_size, img_size))
            img = np.moveaxis(img, -1, 0) # (batch_size, width, heigth, channels) -> (batch_size, channels, width, heigth)
            img_list.append(img/255.0) # Normalize the image

        outputs = ort_session.run(None, {"input":img_list})
        out = np.array(outputs)

        for image_num, image_name in zip(range(out.shape[1]), os.listdir(image_path)):
            index = out[0][image_num]
            print("Image : {0}, Class : {1}".format(image_name, labels[np.argmax(index)]))

    except Exception as e:
        print("Exception occured : ", e)

    

def arguement_parser():
    parser = argparse.ArgumentParser(description="Parse input for model training")

    parser.add_argument('--model_path', type=str, default="/home/sahil/Documents/Classifiers/Timm_Pipeline/classifier.onnx", help="PyTorch model path")
    parser.add_argument('--image_path', type=str, default="/home/sahil/Documents/Classifiers/sample_data", help='Path to images')
    parser.add_argument('--labels', type=str, default="buildings, forests, mountains, glacier, street, sea", help='labels')
    parser.add_argument('--batch_size', type=int, default=4, help='batch size for inference')
    parser.add_argument('--img_size', type=int, default=224, help='Input image size')
    parser.add_argument('--target_size', type=int, default=6, help='Number of classes')

    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = arguement_parser()
    main(args)
