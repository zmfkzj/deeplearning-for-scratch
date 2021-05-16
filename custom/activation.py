import numpy as np

class Activation:

    @staticmethod
    def relu(x:np.ndarray):
        return np.maximum(0,x)
    
    @staticmethod
    def relu_grad(x):
        grad = np.zeros(x)
        grad[x>=0] = 1
        return grad

    @staticmethod
    def identify(x):
        return x

    @staticmethod
    def identify_grad(x):
        return 1

    @staticmethod
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))    

    def sigmoid_grad(self, x):
        return (1.0 - self.sigmoid(x)) * self.sigmoid(x)