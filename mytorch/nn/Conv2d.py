import numpy as np
from resampling import *


class Conv2d_stride1():
    def __init__(self, in_channels, out_channels,
                 kernel_size, weight_init_fn=None, bias_init_fn=None):

        # Do not modify this method

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size

        if weight_init_fn is None:
            self.W = np.random.normal(
                0, 1.0, (out_channels, in_channels, kernel_size, kernel_size))
        else:
            self.W = weight_init_fn(
                out_channels,
                in_channels,
                kernel_size,
                kernel_size)

        if bias_init_fn is None:
            self.b = np.zeros(out_channels)
        else:
            self.b = bias_init_fn(out_channels)

        self.dLdW = np.zeros(self.W.shape)
        self.dLdb = np.zeros(self.b.shape)

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, input_height, input_width)
        Return:
            Z (np.array): (batch_size, out_channels, output_height, output_width)
        """
        self.A = A
        self.batch_size, self.in_channels, self.input_height, self.input_width = A.shape
        self.output_height = self.input_height - self.kernel_size + 1
        self.output_width = self.input_width - self.kernel_size + 1

        Z = np.zeros((self.batch_size, self.out_channels,self.output_height, self.output_width), dtype=A.dtype)

        for i in range(self.output_height):
            for j in range(self.output_width):
                A_slice = A[:, :, i:i + self.kernel_size, j:j + self.kernel_size]
                Z[:, :, i, j] = np.tensordot(A_slice, self.W, axes=([1, 2, 3], [1, 2, 3])) + self.b

        return Z

    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch_size, out_channels, output_height, output_width)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_height, input_width)
        """

        self.batch_size, self.out_channels, self.output_height, self.output_width = dLdZ.shape
        self.batch_size, self.in_channels, self.input_height, self.input_width = self.A.shape

        self.dLdW = np.zeros_like(self.W)
        for i in range(self.output_height):
            for j in range(self.output_width):
                A_slice = self.A[:, :, i:i + self.kernel_size, j:j + self.kernel_size]
                dLdZ_slice = dLdZ[:, :, i, j]
                self.dLdW += np.tensordot(dLdZ_slice, A_slice, axes=([0], [0]))
        self.dLdb = np.sum(dLdZ, axis=(0, 2, 3))
        pad_width = self.kernel_size - 1
        dLdZ_padded = np.pad(dLdZ, ((0, 0), (0, 0), (pad_width, pad_width), (pad_width, pad_width)), mode='constant')
        W_flipped = np.flip(self.W, axis=(2, 3))
        dLdA = np.zeros((self.batch_size, self.in_channels, self.input_height, self.input_width))
        for i in range(self.input_height):
            for j in range(self.input_width):
                dLdZ_slice = dLdZ_padded[:, :, i:i + self.kernel_size, j:j + self.kernel_size]
                dLdA[:, :, i, j] = np.tensordot(dLdZ_slice, W_flipped, axes=([1, 2, 3], [0, 2, 3]))

        return dLdA


class Conv2d():
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding=0,
                 weight_init_fn=None, bias_init_fn=None):
        # Do not modify the variable names
        self.stride = stride
        self.pad = padding

        # Initialize Conv2d() and Downsample2d() isntance
        self.conv2d_stride1 = Conv2d_stride1(in_channels, out_channels, kernel_size, weight_init_fn, bias_init_fn)
        self.downsample2d = Downsample2d(stride)

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, input_height, input_width)
        Return:
            Z (np.array): (batch_size, out_channels, output_height, output_width)
        """
        
        # Pad the input appropriately using np.pad() function
        A_padded = np.pad(A, ((0, 0), (0, 0), (self.pad, self.pad), (self.pad, self.pad)), mode='constant')

        # Call Conv2d_stride1
        Z = self.conv2d_stride1.forward(A_padded)

        # downsample
        Z = self.downsample2d.forward(Z)

        return Z

    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch_size, out_channels, output_height, output_width)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_height, input_width)
        """

        # Call downsample1d backward
        dLdZ = self.downsample2d.backward(dLdZ)

        # Call Conv1d_stride1 backward
        dLdA = self.conv2d_stride1.backward(dLdZ)

        # Unpad the gradient
        dLdA = dLdA[:, :, self.pad:-self.pad, self.pad:-self.pad]

        return dLdA
