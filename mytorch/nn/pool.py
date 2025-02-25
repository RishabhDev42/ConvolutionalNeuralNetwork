import numpy as np

from resampling import *


class MaxPool2d_stride1():

    def __init__(self, kernel):
        self.kernel = kernel

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, input_width, input_height)
        Return:
            Z (np.array): (batch_size, out_channels, output_width, output_height)
        """
        self.A = A
        self.batch_size, self.in_channels, self.input_width, self.input_height = A.shape
        self.output_width = self.input_width - self.kernel + 1
        self.output_height = self.input_height - self.kernel + 1
        Z = np.zeros((self.batch_size, self.in_channels, self.output_width, self.output_height), dtype=A.dtype)
        for i in range(self.output_width):
            for j in range(self.output_height):
                A_slice = A[:, :, i:i + self.kernel, j:j + self.kernel]
                Z[:, :, i, j] = np.max(A_slice, axis=(2, 3))
        return Z

    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch_size, out_channels, output_width, output_height)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_width, input_height)
        """
        dLdA = np.zeros((self.batch_size, self.in_channels, self.input_width, self.input_height), dtype=dLdZ.dtype)
        for i in range(self.output_width):
            for j in range(self.output_height):
                A_slice = self.A[:, :, i:i + self.kernel, j:j + self.kernel]
                max_mask = (A_slice == np.max(A_slice, axis=(2, 3), keepdims=True))
                dLdA[:, :, i:i + self.kernel, j:j + self.kernel] += max_mask * dLdZ[:, :, i, j][:, :, None, None]
        return dLdA


class MeanPool2d_stride1():

    def __init__(self, kernel):
        self.kernel = kernel

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, input_width, input_height)
        Return:
            Z (np.array): (batch_size, out_channels, output_width, output_height)
        """
        self.batch_size, self.in_channels, self.input_width, self.input_height = A.shape
        self.output_width = self.input_width - self.kernel + 1
        self.output_height = self.input_height - self.kernel + 1
        Z = np.zeros((self.batch_size, self.in_channels, self.output_width, self.output_height), dtype=A.dtype)
        for i in range(self.output_width):
            for j in range(self.output_height):
                A_slice = A[:, :, i:i + self.kernel, j:j + self.kernel]
                Z[:, :, i, j] = np.mean(A_slice, axis=(2, 3))
        return Z

    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch_size, out_channels, output_width, output_height)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_width, input_height)
        """

        dLdA = np.zeros((self.batch_size, self.in_channels, self.input_width, self.input_height), dtype=dLdZ.dtype)
        for i in range(self.output_width):
            for j in range(self.output_height):
                dLdA[:, :, i:i + self.kernel, j:j + self.kernel] += dLdZ[:, :, i, j][:, :, None, None] / (self.kernel ** 2)
        return dLdA


class MaxPool2d():

    def __init__(self, kernel, stride):
        self.kernel = kernel
        self.stride = stride

        # Create an instance of MaxPool2d_stride1
        self.maxpool2d_stride1 =  MaxPool2d_stride1(kernel)
        self.downsample2d = Downsample2d(stride)

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, input_width, input_height)
        Return:
            Z (np.array): (batch_size, out_channels, output_width, output_height)
        """
        poolOutput = self.maxpool2d_stride1.forward(A)
        Z = self.downsample2d.forward(poolOutput)

        return Z

    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch_size, out_channels, output_width, output_height)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_width, input_height)
        """
        downsampled_dLdZ = self.downsample2d.backward(dLdZ)
        dLdA = self.maxpool2d_stride1.backward(downsampled_dLdZ)

        return dLdA


class MeanPool2d():

    def __init__(self, kernel, stride):
        self.kernel = kernel
        self.stride = stride

        # Create an instance of MaxPool2d_stride1
        self.meanpool2d_stride1 = MeanPool2d_stride1(kernel)
        self.downsample2d = Downsample2d(stride)

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, input_width, input_height)
        Return:
            Z (np.array): (batch_size, out_channels, output_width, output_height)
        """
        poolOutput = self.meanpool2d_stride1.forward(A)
        Z = self.downsample2d.forward(poolOutput)

        return Z

    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch_size, out_channels, output_width, output_height)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_width, input_height)
        """
        downsampled_dLdZ= self.downsample2d.backward(dLdZ)
        dLdA = self.meanpool2d_stride1.backward(downsampled_dLdZ)
        return dLdA
