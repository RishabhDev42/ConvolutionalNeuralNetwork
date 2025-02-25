# DO NOT import any additional 3rd party external libraries as they will not
# be available to AutoLab and are not needed (or allowed)
import sys

from flatten import Flatten
from Conv1d import Conv1d
from activation import ReLU

sys.path.append('mytorch')


class CNN_SimpleScanningMLP():
    def __init__(self):
        # Your code goes here -->
        # self.conv1 = ???
        # self.conv2 = ???
        # self.conv3 = ???
        # ...
        # <---------------------
        self.conv1 = Conv1d(24,8,8,4)
        self.conv2 = Conv1d(8,16,1,1)
        self.conv3 = Conv1d(16,4,1,1)
        self.layers = [self.conv1,ReLU(),self.conv2,ReLU(),self.conv3,Flatten()] # TODO: Add the layers in the correct order

    def init_weights(self, weights):
        """
        Args:
            w1 (np.array): (kernel_size * in_channels, out_channels)
            w2 (np.array): (kernel_size * in_channels, out_channels)
            w3 (np.array): (kernel_size * in_channels, out_channels)
        """
        
        w1, w2, w3 = weights[0], weights[1], weights[2]
        # Load the weights for your CNN from the MLP Weights given
        # w1, w2, w3 contain the weights for the three layers of the MLP
        # Load them appropriately into the CNN
        
        # TODO: For each weight:
        #   1 : Transpose the layer's weight matrix
        #   2 : Reshape the weight into (out_channels, kernel_size, in_channels)
        #   3 : Transpose weight back into (out_channels, in_channels, kernel_size)

        self.conv1.conv1d_stride1.W = w1.transpose().reshape(8, 8, 24).transpose([0, 2, 1])
        self.conv2.conv1d_stride1.W = w2.transpose().reshape(16, 1, 8).transpose([0, 2, 1])
        self.conv3.conv1d_stride1.W = w3.transpose().reshape(4, 1, 16).transpose([0, 2, 1])

    def forward(self, A):
        """
        Do not modify this method

        Argument:
            A (np.array): (batch size, in channel, in width)
        Return:
            Z (np.array): (batch size, out channel , out width)
        """

        Z = A
        for layer in self.layers:
            Z = layer.forward(Z)
        return Z

    def backward(self, dLdZ):
        """
        Do not modify this method

        Argument:
            dLdZ (np.array): (batch size, out channel, out width)
        Return:
            dLdA (np.array): (batch size, in channel, in width)
        """

        for layer in self.layers[::-1]:
            dLdA = layer.backward(dLdA)
        return dLdA


class CNN_DistributedScanningMLP():
    def __init__(self):
        # Your code goes here -->
        # self.conv1 = ???
        # self.conv2 = ???
        # self.conv3 = ???
        # ...
        # <---------------------
        self.conv1 = Conv1d(24, 2, 2, 2)
        self.conv2 = Conv1d(2, 8, 2, 2)
        self.conv3 = Conv1d(8,4, 2, 1)
        self.layers = [self.conv1,ReLU(),self.conv2,ReLU(),self.conv3,Flatten()] # TODO: Add the layers in the correct order

    def __call__(self, A):
        # Do not modify this method
        return self.forward(A)

    def init_weights(self, weights):
        """
        Args:
            w1 (np.array): (kernel_size * in_channels, out_channels)
            w2 (np.array): (kernel_size * in_channels, out_channels)
            w3 (np.array): (kernel_size * in_channels, out_channels)
        """
        
        w1, w2, w3 = weights[0], weights[1], weights[2]
        # Load the weights for your CNN from the MLP Weights given
        # w1, w2, w3 contain the weights for the three layers of the MLP
        # Load them appropriately into the CNN

        # TODO: For each weight:
        #   1 : Transpose the layer's weight matrix
        #   2 : Reshape the weight into (out_channels, kernel_size, in_channels)
        #   3 : Transpose weight back into (out_channels, in_channels, kernel_size)
        #   4 : Slice the weight matrix and reduce to only the shared weights
        #   (hint: be careful, steps 1-3 are similar, but not exactly like in the simple scanning MLP)

        self.conv1.conv1d_stride1.W = w1[:2*24,:2].reshape(2, 24, 2).transpose([2, 1, 0])
        self.conv2.conv1d_stride1.W = w2[:2*2,:8].reshape(2, 2, 8).transpose([2, 1, 0])
        self.conv3.conv1d_stride1.W = w3[:2*8,:4].reshape(2, 8, 4).transpose([2, 1, 0])

    def forward(self, A):
        """
        Do not modify this method

        Argument:
            A (np.array): (batch size, in channel, in width)
        Return:
            Z (np.array): (batch size, out channel , out width)
        """

        Z = A
        for layer in self.layers:
            Z = layer.forward(Z)
        return Z

    def backward(self, dLdZ):
        """
        Do not modify this method

        Argument:
            dLdZ (np.array): (batch size, out channel, out width)
        Return:
            dLdA (np.array): (batch size, in channel, in width)
        """
        dLdA = dLdZ
        for layer in self.layers[::-1]:
            dLdA = layer.backward(dLdA)
        return dLdA
