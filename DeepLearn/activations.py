import numpy as np

def Sigmoid(z):
    """The sigmoid activation function."""
    return 1.0/(1.0 + np.exp(-z))

def ReLU(z):
    """The ReLU (rectified linear unit) activation function."""
    return np.maximum(0, z)

class SoftMax():
    """This is the SoftMax activation function. It is creates a probability distribution from the values of row vector and is
    especially useful for classification when the target labels are one-hot encoded. Overflow due to exponentation is prevented by
    subtracting the max value of a sample from all its values."""

    def __init__(self):
        self.output = None

    def forward(self, z_array):

        # Overflow prevention,
        z_array -= np.max(z_array, axis = 1, keepdims = True)
        
        self.output = np.exp(z_array) / np.sum(np.exp(z_array), axis = 1, keepdims = True)
        return self.output
