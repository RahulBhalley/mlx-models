# Copyright ©️ 2024 Rahul Bhalley

import mlx.core as mx
import mlx.nn as nn
from mlx.nn.layers.base import Module

from .pooling import AdaptiveAvgPool2d

import json
import os
import sys
import subprocess


class VGG19(Module):

    def __init__(self, load_weights=False):
        super().__init__(),

        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),#, dilation=1),#, ceil_mode=False),
            nn.Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),#, dilation=1),#, ceil_mode=False),
            nn.Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),#, dilation=1),#, ceil_mode=False),
            nn.Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),#, dilation=1),#, ceil_mode=False),
            nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),#, dilation=1),#, ceil_mode=False),
        )
        self.avgpool = AdaptiveAvgPool2d(output_size=(7, 7))
        self.classifier = nn.Sequential(
            nn.Linear(25088, 4096, bias=True),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(4096, 4096, bias=True),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(4096, 1000, bias=True),
        )

        if load_weights:
            # Get model and script path
            model_path = self.get_pretrained_model_path("vgg19")
            script_path = os.path.join(os.path.dirname(__file__), 'convert_weights_from_torch_to_mlx.py')
            
            # Time to download/convert model
            if model_path == None or not os.path.exists(model_path):
                print(f"Pretrained model not found at {model_path}. Trying to download...")
                subprocess.run(f"{sys.executable} {script_path} vgg19", shell=True, check=True)
            else:
                self.load_weights(model_path)
                print("Loaded pretrained weights")
    

    def __call__(self, a: mx.array) -> mx.array:
        a = self.features(a)
        a = self.avgpool(a)
        a = mx.reshape(a, (a.shape[0], -1))
        a = self.classifier(a)
        return a
    
    def get_pretrained_model_path(self, model_name):
        mappings_file = os.path.expanduser('~/.cache/mlx.models/model_mappings.json')
        if not os.path.exists(mappings_file):
            print(f"Model mappings file not found at {mappings_file}")
            return None
        with open(mappings_file, 'r') as f:
            mappings = json.load(f)
        return mappings.get(model_name)
