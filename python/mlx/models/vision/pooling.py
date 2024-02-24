# Copyright © 2023 Apple Inc.


from typing import Callable, List, Optional, Tuple, Union

import mlx.core as mx
from mlx.nn.layers.base import Module

import math

__all__ = [
    "Pool1d",
    "Pool2d",
    "MaxPool1d",
    "AvgPool1d",
    "MaxPool2d",
    "AvgPool2d",
    "AdaptiveAvgPool2d",
]

class PoolBase(Module):
    """
    Base class for pooling layers.

    Args:
        kernel_size (int or tuple of int): The size of the window for each dimension of the input tensor.
        stride (int or tuple of int, optional): The stride of the window. Default value is `kernel_size`.
        padding (int or tuple of int): Implicit zero padding to be added on both sides for each dimension.

    Attributes:
        kernel_size: The size of the window for each dimension of the input tensor.
        stride: The stride of the window.
        padding: The padding value for each dimension of the input tensor.
    """

    def __init__(
        self,
        kernel_size: Union[int, Tuple[int, int]],
        stride: Optional[Union[int, Tuple[int, int]]] = None,
        padding: Union[int, Tuple[int, int]] = 0,
    ):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride if stride is not None else kernel_size
        self.padding = padding

    def _get_padding(self, features_sizes: List[int]) -> List[Tuple[int, int]]:
        if isinstance(self.padding, int):
            return [(0, 0)] + [(self.padding, self.padding)] * len(features_sizes) + [(0, 0)]
        if len(self.padding) != len(features_sizes):
            raise ValueError("The number of provided padding values must match the number of feature axes.")
        return [(0, 0)] + [(p, p) for p in self.padding] + [(0, 0)]

    def _get_stride(self, features_sizes: List[int]) -> List[int]:
        if isinstance(self.stride, int):
            return [self.stride] * len(features_sizes)
        if len(self.stride) != len(features_sizes):
            raise ValueError("The number of provided strides must match the number of feature axes.")
        return self.stride

    def _get_kernel_size(self, features_sizes: List[int]) -> List[int]:
        if isinstance(self.kernel_size, int):
            return [self.kernel_size] * len(features_sizes)
        if len(self.kernel_size) != len(features_sizes):
            raise ValueError("kernel_size must match the number of feature axes.")
        return self.kernel_size

    def _get_row_contiguous_strides(self, a: mx.array) -> List[int]:
        return list(reversed(mx.cumprod(mx.array([1] + list(reversed(a.shape))))[:-1].tolist()))

    def _pad(self, a: mx.array, features_sizes: List[int], padding_value: float) -> mx.array:
        return mx.pad(a, self._get_padding(features_sizes), padding_value)

    def _extra_repr(self):
        return f"{self.kernel_size}, stride={self.stride}, padding={self.padding}"


class Pool1d(PoolBase):
    def __init__(
        self, kernel_size: int, stride: Optional[int] = None, padding: Optional[int] = 0
    ):
        super().__init__(kernel_size, stride, padding)

    def __call__(
        self,
        a: mx.array,
        pooling_operator: Callable[[mx.array, List[int]], mx.array],
        padding_value: float,
    ) -> mx.array:
        if a.ndim != 3:
            raise ValueError("[pooling] the input must be three-dimensional.")
        # Pad if necessary
        if self.padding != 0:
            _, input_feature_size, _ = a.shape
            a = self._pad(a, [input_feature_size], padding_value)
        # Assumes a.shape = (batch_size, sequence_length, num_channels)
        [batch_size, sequence_length, num_channels] = a.shape
        [
            batch_stride,
            sequence_length_stride,
            channels_stride,
        ] = self._get_row_contiguous_strides(a)
        [kernel_size] = self._get_kernel_size([sequence_length])
        [pool_stride] = self._get_stride([sequence_length])
        # Compute windows : [batch_size, output_dim, kernel_size, num_channels]
        windows = mx.as_strided(
            a,
            shape=(
                batch_size,
                (sequence_length - kernel_size) // pool_stride + 1,
                kernel_size,
                num_channels,
            ),
            strides=(
                batch_stride,
                pool_stride * sequence_length_stride,
                sequence_length_stride,
                channels_stride,
            ),
        )
        # Reduce over windows
        return pooling_operator(windows, 2)


class Pool2d(PoolBase):
    """
    Base class for 2D pooling operations with dilation support.

    This class extends `PoolBase` by adding dilation support and a method for performing the pooling operation.

    Args:
        kernel_size (int or tuple of int): The size of the pooling window.
        stride (int or tuple of int, optional): The stride of the pooling window. Defaults to `kernel_size`.
        padding (int or tuple of int, optional): Zero-padding added to both sides of the input. Default is 0.
        dilation (int or tuple of int, optional): The spacing between kernel elements. Default is 1.

    Shape:
        - Input: (N, H, W, C), where N is the batch size, H is the height, W is the width, and C is the number of channels.
        - Output: (N, H_out, W_out, C), where H_out and W_out depend on the kernel size, stride, padding, and dilation.
    """

    def __init__(
        self,
        kernel_size: Union[int, Tuple[int, int]],
        stride: Optional[Union[int, Tuple[int, int]]] = None,
        padding: Union[int, Tuple[int, int]] = 0,
        dilation: Optional[Union[int, Tuple[int, int]]] = 1,
    ):
        super().__init__(kernel_size, stride, padding)
        self.dilation = dilation if isinstance(dilation, tuple) else (dilation, dilation)

    def __call__(
        self,
        a: mx.array,
        pooling_operator: Callable[[mx.array, List[int]], mx.array],
        padding_value: float,
    ) -> mx.array:
        if a.ndim != 4:
            raise ValueError("[pooling] the input must be four-dimensional.")

        # Pad if necessary
        if self.padding != 0:
            _, input_height, input_width, _ = a.shape
            a = self._pad(a, [input_height, input_width], padding_value)

        [batch_size, height, width, num_channels] = a.shape
        [batch_stride, height_stride, width_stride, channels_stride] = self._get_row_contiguous_strides(a)

        [kernel_height, kernel_width] = self._get_kernel_size([height, width])
        [pool_height_stride, pool_width_stride] = self._get_stride([height, width])

        # Adjust kernel size for dilation
        dilated_kernel_height = kernel_height + (kernel_height - 1) * (self.dilation[0] - 1)
        dilated_kernel_width = kernel_width + (kernel_width - 1) * (self.dilation[1] - 1)

        windows = mx.as_strided(
            a,
            shape=(
                batch_size,
                (height - dilated_kernel_height) // pool_height_stride + 1,
                (width - dilated_kernel_width) // pool_width_stride + 1,
                kernel_height,
                kernel_width,
                num_channels,
            ),
            strides=(
                batch_stride,
                pool_height_stride * height_stride,
                pool_width_stride * width_stride,
                self.dilation[0] * height_stride,
                self.dilation[1] * width_stride,
                channels_stride,
            ),
        )

        # Reduce over windows
        return pooling_operator(windows, (3, 4))


class MaxPool1d(Pool1d):
    r"""Applies a 1-dimensional max pooling.

    The channels are expected to be last i.e. the input shape should be :math:`(N, L, C)` where:
        - ``N`` is the batch dimension
        - ``L`` is the sequence length
        - ``C`` is the number of input channels

    Assuming :attr:`kernel_size` is :math:`kL`, the output is a tensor of shape :math:`(N, L_{out}, C)`, given by:
        .. math::
            \text{out}(N_i, k, C_j) = \max_{m=0, \ldots, kL - 1}
                    \text{input}(N_i, \text{stride} \times k + m, C_j),
    where :math:`L_{out} = \left\lfloor \frac{L + 2 \times \text{padding} - \text{kernel_size}}{\text{stride}}\right\rfloor + 1`.

    Args:
        kernel_size (int): The size of the pooling window kernel.
        stride (int, optional): The stride of the pooling window.
            Default: :attr:`kernel_size`.
        padding (int, optional): How many positions to 0-pad the input with. The padding amount is applied to both sides of the feature axis.
            Default: 0.

    Shape:
        - Input: :math:`(N, L, C)`.
        - Output: :math:`(N, L_{out}, C)`.

    Examples:
        >>> import mlx.core as mx
        >>> import mlx.nn.layers as nn
        >>> x = mx.array([[[0, 1, 2],
                           [3, 4, 5],
                           [6, 7, 8],
                           [9, 10, 11]],
                          [[12, 13, 14],
                           [15, 16, 17],
                           [18, 19, 20],
                           [21, 22, 23]]])
        >>> pool = nn.MaxPool1d(kernel_size=2, stride=2)
        >>> pool(x)
    """

    def __init__(
        self, kernel_size: int, stride: Optional[int] = None, padding: Optional[int] = 0
    ):
        super().__init__(kernel_size, stride, padding)

    def __call__(self, a: mx.array) -> mx.array:
        return super().__call__(a, mx.max, float("-inf"))


class AvgPool1d(Pool1d):
    r"""Applies a 1-dimensional average pooling.

    The channels are expected to be last i.e. the input shape should be :math:`(N, L, C)` where:
        - ``N`` is the batch dimension
        - ``L`` is the sequence length
        - ``C`` is the number of input channels

    Assuming :attr:`kernel_size` is :math:`kL`, the output is a tensor of shape :math:`(N, L_{out}, C)`, given by:
        .. math::
            \text{out}(N_i, k, C_j) = \frac{1}{kL} \sum_{m=0, \ldots, kL - 1}
                    \text{input}(N_i, \text{stride} \times k + m, C_j),
    where :math:`L_{out} = \left\lfloor \frac{L + 2 \times \text{padding} - \text{kernel_size}}{\text{stride}}\right\rfloor + 1`.

    Args:
        kernel_size (int): The size of the pooling window kernel.
        stride (int, optional): The stride of the pooling window.
            Default: :attr:`kernel_size`.
        padding (int, optional): How many positions to pad the input with. The padding value is ``float("-inf")`` and it is applied to both sides of the sequence length axis.
            Padded values are included in the average pooling calculation.
            Default: 0.

    Shape:
        - Input: :math:`(N, L, C)`.
        - Output: :math:`(N, L_{out}, C)`.

    Examples:
        >>> import mlx.core as mx
        >>> import mlx.nn.layers as nn
        >>> x = mx.array([[[0, 1, 2],
                           [3, 4, 5],
                           [6, 7, 8],
                           [9, 10, 11]],
                          [[12, 13, 14],
                           [15, 16, 17],
                           [18, 19, 20],
                           [21, 22, 23]]])
        >>> pool = nn.AvgPool1d(kernel_size=2, stride=2)
        >>> pool(x)
    """

    def __init__(
        self, kernel_size: int, stride: Optional[int] = None, padding: Optional[int] = 0
    ):
        super().__init__(kernel_size, stride, padding)

    def __call__(self, a: mx.array) -> mx.array:
        return super().__call__(a, mx.mean, 0)


class MaxPool2d(Pool2d):
    r"""
    Applies a 2D max pooling over an input signal composed of several input planes.

    The parameters `kernel_size`, `stride`, `padding`, and `dilation` can be either a single int or a tuple of two ints.

    Args:
        kernel_size (int or tuple of int): The size of the window to take a max over.
        stride (int or tuple of int, optional): The stride of the window. Default value is `kernel_size`.
        padding (int or tuple of int, optional): Implicit zero padding to be added on both sides.
        dilation (int or tuple of int, optional): The spacing between kernel elements. Default is 1.

    Shape:
        - Input: (N, H, W, C), where N is the batch size, H is the height, W is the width, and C is the number of channels.
        - Output: (N, H_out, W_out, C), where H_out and W_out are calculated based on the kernel size, stride, padding, and dilation.

    Examples:
        >>> import mlx.core as mx
        >>> import mlx.nn.layers as nn
        >>> x = mx.array([[[[0, 1],
        ...                 [2, 3],
        ...                 [4, 5],
        ...                 [6, 7]],
        ...                [[8, 9],
        ...                 [10, 11],
        ...                 [12, 13],
        ...                 [14, 15]],
        ...                [[16, 17],
        ...                 [18, 19],
        ...                 [20, 21],
        ...                 [22, 23]],
        ...                [[24, 25],
        ...                 [26, 27],
        ...                 [28, 29],
        ...                 [30, 31]]]])
        >>> pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1)
        >>> pool(x)
    """

    def __init__(
        self,
        kernel_size: Union[int, Tuple[int, int]],
        stride: Optional[Union[int, Tuple[int, int]]] = None,
        padding: Union[int, Tuple[int, int]] = 0,
        dilation: Optional[Union[int, Tuple[int, int]]] = 1,
    ):
        super().__init__(kernel_size, stride, padding, dilation)

    def __call__(self, a: mx.array) -> mx.array:
        return super().__call__(a, mx.max, float("-inf"))


class AvgPool2d(Pool2d):
    r"""Applies a 2-dimensional average pooling.

    The channels are expected to be last i.e. the input shape should be :math:`(N, H, W, C)` where:
        - ``N`` is the batch dimension
        - ``H`` is the input image height
        - ``W`` is the input image width
        - ``C`` is the number of input channels

    Assuming :attr:`kernel_size` is :math:`(kH, kW)`, the output is a tensor of shape :math:`(N, H_{out}, W_{out}, C)`, given by:

    .. math::
        \begin{aligned}
            \text{out}(N_i, h, w, C_j) = \frac{1} {kH \times kW} & \sum_{m=0, \ldots, kH-1} \sum_{n=0, \ldots, kW-1} \\
                                        & \text{input}(N_i, \text{stride[0]} \times h + m, \text{stride[1]} \times w + n, C_j)
        \end{aligned}
    where :math:`H_{out} = \left\lfloor\frac{H + 2 * \text{padding[0]} - \text{kernel_size[0]}}{\text{stride[0]}}\right\rfloor + 1`,
    :math:`W_{out} = \left\lfloor\frac{W + 2 * \text{padding[1]} - \text{kernel_size[1]}}{\text{stride[1]}}\right\rfloor + 1`.

    The parameters :attr:`kernel_size`, :attr:`stride`, :attr:`padding`, can either be:
        - a single ``int`` -- in which case the same value is used for both the height and width axis;
        - a ``tuple`` of two ``int`` s -- in which case, the first `int` is used for the height axis, the second `int` for the width axis.

    Args:
        kernel_size (int, tuple(int, int)): The size of the pooling window.
        stride (int, tuple(int, int), optional): The stride of the window. Default: :attr:`kernel_size`.
        padding (int, tuple(int, int), optional): The padding amount on both sides of the height and width axis. The padding value is ``float("-inf")`` and it is applied to both sides of the height axis and the width axis. Padded values are included in the average pooling calculation. Default: 0.

    Shape:
        - Input: :math:`(N, H, W, C)`.
        - Output: :math:`(N, H_{out}, W_{out}, C)`.

    Examples: 
        >>> import mlx.core as mx
        >>> import mlx.nn.layers as nn
        >>> x = mx.array([[[[0, 1],
                            [2, 3],
                            [4, 5],
                            [6, 7]],
                           [[8, 9],
                            [10, 11],
                            [12, 13],
                            [14, 15]],
                           [[16, 17],
                            [18, 19],
                            [20, 21],
                            [22, 23]],
                           [[24, 25],
                            [26, 27],
                            [28, 29],
                            [30, 31]]]])
        >>> pool = nn.AvgPool2d(kernel_size=2, stride=2)
        >>> pool(x)    
    """

    def __init__(
        self,
        kernel_size: Union[int, Tuple[int, int]],
        stride: Optional[Union[int, Tuple[int, int]]] = None,
        padding: Optional[Union[int, Tuple[int, int]]] = 0,
    ):
        super().__init__(kernel_size, stride, padding)

    def __call__(self, a: mx.array) -> mx.array:
        return super().__call__(a, mx.mean, 0)
    

class AdaptiveAvgPool2d(Module):
    def __init__(self, output_size=Tuple[int, int]):
        super().__init__()
        self.output_size = output_size

    def __call__(self, a: mx.array) -> mx.array:
        # Calculate the desired output size
        out_h, out_w = self.output_size
        # Get the size of the input feature maps
        n, h, w, c = a.shape

        # Compute the stride and kernel size to cover the entire feature map
        stride_h = int(h / out_h)
        stride_w = int(w / out_w)
        kernel_h = h - int((out_h - 1) * stride_h)
        kernel_w = w - int((out_w - 1) * stride_w)

        # Create average pooling instance using the computed kernel and stride
        avg_pool2d = AvgPool2d(kernel_size=(kernel_h, kernel_w), stride=(stride_h, stride_w))

        # Apply adaptive average pooling using the calculated kernel size and stride
        # Here, the floor function is used to ensure integer values for kernel and stride sizes
        a = avg_pool2d(a)
        return a
