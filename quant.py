#! /usr/bin/env python3
# Adapted from https://github.com/microsoft/onnxruntime-inference-examples/blob/d031f879c9a8d33c8b7dc52c5bc65fe8b9e3960d/quantization/image_classification/cpu/run.py

import os
import sys
import numpy as np
import re
import abc
import subprocess
import json
import time
from PIL import Image

import onnx
import onnxruntime
from onnx import helper, TensorProto, numpy_helper
from onnxruntime.quantization import quantize_static, CalibrationDataReader, QuantFormat, QuantType

import modman
import dataman as dm


import argparse
parser = argparse.ArgumentParser()
parser.add_argument("model_name", choices=['resnet50', 'vgg16_bn', 'densenet121', 'googlenet', 'inception_v3'])
parser.add_argument("output_root_dir", help="root output directory")
parser.add_argument("--quant-format",
                    default=QuantFormat.QOperator,
                    type=QuantFormat.from_string,
                    choices=list(QuantFormat))
parser.add_argument("--per-channel", default=False, type=bool)
parser.add_argument('--image-size', default=32)
args = parser.parse_args()


class DataReader(CalibrationDataReader):
    def __init__(self, dataset, image_size, batch_size):
        self.loader = dm.get_benign_loader(dataset, image_size, 'train', batch_size)
        self.loader_iter = iter(self.loader)

    def get_next(self):
        try:
            return {'input0': next(self.loader_iter)[0].numpy()}
        except StopIteration:
            return None


def benchmark(model_path, batch_size):
    session = onnxruntime.InferenceSession(model_path)
    input_name = session.get_inputs()[0].name

    total = 0.0
    runs = 10
    input_data = np.zeros((batch_size, 3, args.image_size, args.image_size), np.float32)
    # Warming up
    _ = session.run([], {input_name: input_data})
    for i in range(runs):
        start = time.perf_counter()
        _ = session.run([], {input_name: input_data})
        end = (time.perf_counter() - start) * 1000
        total += end
        print(f"{end:.2f}ms")
    total /= runs
    print(f"Avg: {total:.2f}ms")


def main():
    dataset = 'CIFAR10'
    model_name = args.model_name
    image_size = args.image_size

    for batch_size in [1, 25]:
        output_root_dir = args.output_root_dir
        output_dir = f'{output_root_dir}/{dataset}/Q{model_name}'
        output_model_path = f'{output_dir}/Q{model_name}-{batch_size}.onnx'
        os.makedirs(output_dir, exist_ok=True)

        tmp_onnx_file = f'./onnx-export/tmp-{dataset}-{model_name}.onnx'
        torch_mod = modman.get_torch_mod(model_name, dataset)
        modman.export_torch_mod(torch_mod, (batch_size, 3, image_size, image_size), tmp_onnx_file, optimise=True)

        dr = DataReader(dataset, image_size, batch_size)
        quantize_static(tmp_onnx_file,
                        output_model_path,
                        dr,
                        quant_format=args.quant_format,
                        per_channel=args.per_channel,
                        weight_type=QuantType.QInt8)
        print('Calibrated and quantized model saved.')

        print('benchmarking fp model...')
        benchmark(tmp_onnx_file, batch_size)

        print('benchmarking int model...')
        benchmark(output_model_path, batch_size)


if __name__ == '__main__':
    main()
