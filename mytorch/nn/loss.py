import numpy as np

from activation import Softmax


class MSELoss:
    def forward(self, A, Y):
        """
        Calculate the Mean Squared error (MSE)
        :param A: Output of the model of shape (N, C)
        :param Y: Ground-truth values of shape (N, C)
        :Return: MSE Loss (scalar)

        Read the writeup (Hint: MSE Loss Section) for implementation details for below code snippet.
        """
        self.A = A
        self.Y = Y
        self.N = self.Y.shape[0]  # TODO
        self.C = self.Y.shape[1]  # TODO
        se = (self.A-self.Y) * (self.A-self.Y) # TODO
        sse = np.dot(np.dot(np.transpose(np.ones((self.N,1))),se),np.ones((self.C,1)))  # TODO
        mse = sse/(np.dot(self.N,self.C))  # TODO
        return mse
        raise NotImplemented  # TODO - What should be the return value?

    def backward(self):
        """
        Calculate the gradient of MSE Loss wrt model output A.
        :Return: Gradient of loss L wrt model output A.

        Read the writeup (Hint: MSE Loss Section) for implementation details for below code snippet.
        """
        dLdA = 2*(self.A-self.Y)/(np.dot(self.N,self.C))
        return dLdA # TODO
        raise NotImplemented  # TODO - What should be the return value?


class CrossEntropyLoss:
    def forward(self, A, Y):
        """
        Calculate the Cross Entropy Loss (XENT)
        :param A: Output of the model of shape (N, C)
        :param Y: Ground-truth values of shape (N, C)
        :Return: CrossEntropyLoss (scalar)

        Read the writeup (Hint: Cross-Entropy Loss Section) for implementation details for below code snippet.
        Hint: Read the writeup to determine the shapes of all the variables.
        Note: Use dtype ='f' whenever initializing with np.zeros()
        """
        self.A = A
        self.Y = Y
        self.N = self.Y.shape[0]  # TODO
        self.C = self.Y.shape[1]  # TODO

        Ones_C = np.ones((self.C,1))  # TODO
        Ones_N = np.ones((self.N,1))  # TODO

        softmax=Softmax()
        self.softmax = softmax.forward(A) # TODO - Can you reuse your own softmax here, if not rewrite the softmax forward logic?

        crossentropy = np.dot((-Y*np.log(self.softmax)),Ones_C)  # TODO
        sum_crossentropy_loss = np.dot(np.transpose(Ones_N),crossentropy)  # TODO
        mean_crossentropy_loss = sum_crossentropy_loss / self.N

        return mean_crossentropy_loss

        raise NotImplemented  # TODO - What should be the return value?

    def backward(self):
        """
        Calculate the gradient of Cross-Entropy Loss wrt model output A.
        :Return: Gradient of loss L wrt model output A.

        Read the writeup (Hint: Cross-Entropy Loss Section) for implementation details for below code snippet.
        """
        dLdA = (self.softmax-self.Y)/self.N  # TODO
        return dLdA
        raise NotImplemented  # TODO - What should be the return value?
