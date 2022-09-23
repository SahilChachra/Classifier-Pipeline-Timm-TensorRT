import tensorrt as trt
import numpy as np
from loguru import logger
import argparse 

"""
Referred from : https://www.github.com/yolov5/blob/master/export.py
"""

def convert2trt(args):
    onnx_model_path = args.onnx_model_path # "/home/jetnano/Documents/Timm Classifier/classifier.onnx"
    workspace = 4
    prefix = "TensorRT : "
    half = args.half
    dynamic = args.dynamic
    input_image_size = args.img_shape # (1, 3, 224, 224)
    f = args.engine_name # "classifier.engine"

    logger.info(f"Converting Onnx model to TensorRT : {trt.__version__}")

    logger_ = trt.Logger(trt.Logger.INFO)
    builder = trt.Builder(logger_)
    config  = builder.create_builder_config()
    config.max_workspace_size = workspace * 1 << 30

    flag = (1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    network = builder.create_network(flag)

    parser = trt.OnnxParser(network, logger_)
    if not parser.parse_from_file(onnx_model_path):
        raise RunttimeError(f"Failed to load ONNX file {onnx_model_path}")

    inputs = [network.get_input(i) for i in range(network.num_inputs)]
    outputs = [network.get_output(i) for i in range(network.num_outputs)]

    for inp in inputs:
        logger.info(f"{prefix} input {inp.name} with shape {inp.shape} {inp.dtype}")
    for out in outputs:
        logger.info(f"{prefix} output {out.name} with shape {out.shape} {out.dtype}")

    if dynamic:
        logger.warning(f"{prefix} requires fixed batch size. Using Batch size 1 for now!")
        profile = builder.create_optimization_profile()
        for inp in inputs:
            profile.set_shape(inp.name, (1, *input_image_size[1:]), (1, *input_image_size[1:]), input_image_size)
        config.add_optimization_profile(profile)

    logger.info(f"{prefix} building FP{16 if half and builder.platform_has_fast_fp16 else 32} engine as {f}")
    if half and builder.platform_has_fast_fp16:
        config.set_flag(trt.BuilderFlag.FP16)
    with builder.build_engine(network, config) as engine, open(f, 'wb') as t:
        t.write(engine.serialize())

def arguement_parser():
    parser = argparse.ArgumentParser(description="Parse input for model training")

    parser.add_argument('--onnx_model_path', type=str, default="/home/jetnano/Documents/Timm Classifier/classifier.onnx", help="Onnx model path")
    parser.add_argument('--img_shape', type=tuple, default=(1, 3, 224, 224), help='Input image shape (Tuple)')
    parser.add_argument('--half', action='store_true', help='Export FP16')
    parser.add_argument('--dynamic', action='store_true', help='Using dynamic batch_size')
    parser.add_argument('--engine_name', type=str, default="classifier.engine", help='Output model name(TRT Engine)')
    
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = arguement_parser()
    convert2trt(args)
