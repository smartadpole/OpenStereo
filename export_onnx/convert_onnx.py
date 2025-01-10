#!/usr/bin/python3 python
# encoding: utf-8
'''
@author: sunhao
@contact: smartadpole@163.com
@file: convert_onnx.py
@time: 2025/1/9 15:56
@desc: 
'''
import sys, os

CURRENT_DIR = os.path.dirname(__file__)
sys.path.append(os.path.join(CURRENT_DIR, '../'))

import os
import argparse
from utils.file import MkdirSimple
from export_onnx.onnx_test import test_dir
from stereo.modeling.models.lightstereo.lightstereo import LightStereo
import torch
from easydict import EasyDict

cfgs_lx = {
    'MAX_DISP': 192,
    'EXPANSE_RATIO': 8,
    'AGGREGATION_BLOCKS': [8, 16, 32],
    'LEFT_ATT': True,
}

cfgs_s = {
    'MAX_DISP': 192,
    'EXPANSE_RATIO': 4,
    'AGGREGATION_BLOCKS': [1, 2, 4],
    'LEFT_ATT': True,
}


def GetArgs():
    parser = argparse.ArgumentParser(description="Export model to ONNX format")
    parser.add_argument("--model", type=str, required=True, help="Path to the trained model.")
    parser.add_argument("-o", "--output", type=str, required=True, help="Path to save the ONNX model.")
    parser.add_argument("--height", help='Model image input height resolution', type=int, default=384)
    parser.add_argument("--width", help='Model image input height resolution', type=int, default=640)
    parser.add_argument("--left_image", type=str, default="", help="test image left file or directory")
    parser.add_argument('--right_image', type=str, default="", help="test image right file or directory")
    parser.add_argument("--test", action="store_true", help="test model")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Device to run the model on.")
    return parser.parse_args()


class WarpModel(torch.nn.Module):
    def __init__(self, config):
        super(WarpModel, self).__init__()
        self.model = LightStereo(config)

    def load(self, model_path, device):
        checkpoint = torch.load(model_path)
        if 'model_state' in checkpoint:
            model_state = checkpoint['model_state']
        else:
            model_state = checkpoint
        self.model.load_state_dict(model_state)
        self.model.to(device)
        self.model.eval()

    def forward(self, image):
        width = image.shape[-1] // 2
        left_img = image[:, :, :, :width]
        right_img = image[:, :, :, width:]
        n, c, h, w = left_img.size()
        data = {
            'left': left_img,
            'right': right_img
        }

        return self.model(data)


if __name__ == "__main__":
    args = GetArgs()
    device = torch.device(args.device)

    model_name = os.path.splitext(os.path.basename(args.model))[0].replace(" ", "_")
    output = os.path.join(args.output, model_name, f'{args.width}_{args.height}')
    onnx_file = os.path.join(output, f'LightStereo_{args.width}_{args.height}_{model_name}_12.onnx')
    MkdirSimple(output)
    output_names = 'output'

    if '-S-' in args.model:
        model = WarpModel(EasyDict(cfgs_s))
    else:
        model = WarpModel(EasyDict(cfgs_lx))
    model.load(args.model, device)

    # Create dummy input for the model
    dummy_input = torch.randn(1, 3, args.height, args.width * 2).to(device)  # Adjust the size as needed

    # Export the depth decoder
    with torch.no_grad():
        torch.onnx.export(model, dummy_input, onnx_file,
                          export_params=True,  # store the trained parameter weights inside the model file
                          opset_version=12,  # the ONNX version to export the model to
                          do_constant_folding=True)

    print("export onnx to {}".format(onnx_file))
    if args.test:
        test_dir(onnx_file, [args.left_image, args.right_image], output)
