import torch
import argparse
import torch.nn as nn
import onnx
from model import CustomModel
from loguru import logger 

def main(args):
    model_path = args.model_path
    img_size = args.img_size
    opset = args.opset
    onnx_name = args.onnx_name
    model_name =args.model_name
    target_size = args.target_size
    device = args.device
    
    if "cuda" in device and torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    # Load model
    logger.info("Loading model...")
    model = CustomModel(model_name=model_name, target_size=target_size, pretrained=False)
    model.load_state_dict(torch.load(model_path, map_location=torch.device(device))["state_dict"])
    model.eval()

    # Setup for export
    batch_size = 1
    x = torch.randn(batch_size, 3, img_size, img_size, requires_grad=True)
    torch_out = model(x)

    try:
        # Export the model
        logger.info("Exporting model...")
        torch.onnx.export(model,               # model being run
                        x,                         # model input (or a tuple for multiple inputs)
                        onnx_name,   # where to save the model (can be a file or file-like object)
                        export_params=True,        # store the trained parameter weights inside the model file
                        opset_version=opset,          # the ONNX version to export the model to
                        do_constant_folding=True,  # whether to execute constant folding for optimization
                        input_names = ['input'],   # the model's input names
                        output_names = ['output'], # the model's output names
                        dynamic_axes={'input' : {0 : 'batch_size'},    # variable length axes
                                        'output' : {0 : 'batch_size'}})

        logger.info("Onnx model export successful!")

        # Test exported model
        logger.info("Checking generated onnx model")
        
        onnx_model = onnx.load(onnx_name)
        
        onnx.checker.check_model(onnx_model)
        logger.info("Onnx model checked!")
        
    except Exception as e:
        logger.info("Exception occured : ", e)

def arguement_parser():
    parser = argparse.ArgumentParser(description="Parse input for model training")

    parser.add_argument('--model_path', type=str, default="/home/sahil/Documents/Classifiers/weight_files/classifier_statedict_ep4_0.937.pt", help="PyTorch model checkpoint path")
    parser.add_argument('--target_size', type=int, default=6, help='Number of classes')
    parser.add_argument('--model_name', type=str, default="resnet50", help="Model name from Timm")
    parser.add_argument('--img_size', type=int, default=224, help='Input image size')
    parser.add_argument('--opset', type=int, default=11, help='Opset value for exporting the model')
    parser.add_argument('--onnx_name', type=str, default="classifier.onnx", help='Output model name(Onnx)')
    parser.add_argument('--device', type=str, default="cuda", help='Device')
    
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = arguement_parser()
    main(args)
