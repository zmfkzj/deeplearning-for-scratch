import sys, os
sys.path.append(os.pardir)  # 부모 디렉터리의 파일을 가져올 수 있도록 설정
import numpy as np
# from dataset.mnist import load_mnist

# (x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, flatten=True, one_hot_label=False)

class Layer:
    def __init__(self, activation='relu') -> None:
        self.activation = activation

    def __call__(self, x):
        self.x = x
        self.input_shape = x.shape
        return getattr(self, self.activation)(self.forward(x))

    def __getattr__(self, name: str):
        if name=='w'or 'b':
            self.init_weight()
        else:
            return self.__getattribute__(name)

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

    def forward(self,x): 
        return getattr(self, self.activation)(x)

    def grad(self,loss):
        return getattr(self, self.activation+'_grad')(loss)
    
    def init_weight(self):pass

class Dense(Layer):
    def __init__(self, node_num, **kwd) -> None:
        super().__init__(self,**kwd)
        self.node_num = node_num
    
    def init_weight(self):
        self.w = np.random.rand(*self.input_shape, self.node_num)
        self.b = np.random.rand(self.input_shape[0], self.node_num)

    def forward(self,x):
        self.h = np.dot(x, self.w) + self.b
        return self.h

    def grad(self, w):
        dw = self.x
        db = 1
        return super().backprop(logit)



if __name__=="__main__":
    x = np.array([1,2,3])
    # relu(x)
    a = Dense(3)(x)
    print()


