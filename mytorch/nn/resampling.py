import math

import numpy as np


class Upsample1d():

    def __init__(self, upsampling_factor):
        self.upsampling_factor = upsampling_factor
        self.batch_size = None
        self.in_channels = None
        self.input_width = None
        self.output_width = None

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, input_width)
        Return:
            Z (np.array): (batch_size, in_channels, output_width)
        """

        # Create a new array Z with the correct shape
        self.batch_size, self.in_channels, self.input_width = A.shape
        self.output_width = self.upsampling_factor * (self.input_width - 1) + 1

        Z = np.zeros((self.batch_size, self.in_channels, self.output_width), A.dtype)

        #  Fill in the values of Z by upsampling A
        Z[:, :, ::self.upsampling_factor] = A

        return Z

    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch_size, in_channels, output_width)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_width)
        """

        #  Slice dLdZ by the upsampling factor to get dLdA

        dLdA = dLdZ[:, :, ::self.upsampling_factor]

        return dLdA


class Downsample1d():

    def __init__(self, downsampling_factor):
        self.downsampling_factor = downsampling_factor
        self.input_width = None
        self.batch_size = None
        self.in_channels = None
        self.output_width = None

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, input_width)
        Return:
            Z (np.array): (batch_size, in_channels, output_width)
        """

        #  Slice A by the downsampling factor to get Z
        # (hint save any other needed information for backprop later)

        self.batch_size, self.in_channels, self.input_width = A.shape
        self.output_width = math.ceil(self.input_width / self.downsampling_factor)

        Z = A[:, :, ::self.downsampling_factor]

        return Z

    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch_size, in_channels, output_width)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_width)
        """

        #  Create a new array dLdA with the correct shape

        dLdA = np.zeros((self.batch_size, self.in_channels, self.input_width), dtype=dLdZ.dtype)

        #  Fill in the values of dLdA with values of A as needed
        dLdA[:, :, ::self.downsampling_factor] = dLdZ

        return dLdA


class Upsample2d():

    def __init__(self, upsampling_factor):
        self.upsampling_factor = upsampling_factor
        self.batch_size = None
        self.in_channels = None
        self.input_height = None
        self.input_width = None
        self.output_height = None
        self.output_width = None

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, input_height, input_width)
        Return:
            Z (np.array): (batch_size, in_channels, output_height, output_width)
        """
        #  Save the input shape
        self.batch_size, self.in_channels, self.input_height, self.input_width = A.shape
        self.output_height = self.upsampling_factor * (self.input_height - 1) + 1
        self.output_width = self.upsampling_factor * (self.input_width - 1) + 1

        #  Create a new array Z with the correct shape

        Z = np.zeros((self.batch_size, self.in_channels, self.output_height, self.output_width), A.dtype)

        # Fill in the values of Z by upsampling A
        Z[:, :, ::self.upsampling_factor, ::self.upsampling_factor] = A

        return Z

    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch_size, in_channels, output_height, output_width)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_height, input_width)
        """

        # Slice dLdZ by the upsampling factor to get dLdA

        dLdA = dLdZ[:, :, ::self.upsampling_factor, ::self.upsampling_factor]

        return dLdA


class Downsample2d():

    def __init__(self, downsampling_factor):
        self.downsampling_factor = downsampling_factor
        self.input_height = None
        self.input_width = None
        self.batch_size = None
        self.in_channels = None
        self.output_height = None
        self.output_width = None

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, input_height, input_width)
        Return:
            Z (np.array): (batch_size, in_channels, output_height, output_width)
        """
        #  Save the input shape
        self.batch_size, self.in_channels, self.input_height, self.input_width = A.shape
        self.output_height = math.ceil(self.input_height / self.downsampling_factor)
        self.output_width = math.ceil(self.input_width / self.downsampling_factor)

        # Slice A by the downsampling factor to get Z
        # (hint save any other needed information for backprop later)

        Z = A[:, :, ::self.downsampling_factor, ::self.downsampling_factor]

        return Z

    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch_size, in_channels, output_height, output_width)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_height, input_width)
        """

        # Create a new array dLdA with the correct shape

        dLdA = np.zeros((self.batch_size, self.in_channels, self.input_height, self.input_width), dtype=dLdZ.dtype)

        # Fill in the values of dLdA with values of A as needed
        dLdA[:, :, ::self.downsampling_factor, ::self.downsampling_factor] = dLdZ

        return dLdA
