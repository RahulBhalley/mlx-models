import mlx.core as mx
import mlx.nn as nn
from mlx.nn.layers.base import Module

# Temporary import till support in ml-explore/mlx
from pooling import AdaptiveAvgPool2d, MaxPool2d


class Inception_V3(Module):

    def __init__(self):
        super().__init__()

    def __call__(self, a: mx.array) -> mx.array:
        return a