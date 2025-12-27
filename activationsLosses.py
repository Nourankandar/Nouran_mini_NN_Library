import numpy as np
from layers import Layer

#activations
class Linear(Layer):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        return x 
    def backward(self, dout):
        return dout 

class Relu(Layer):
    def __init__(self):
        super().__init__()
        self.mask = None
    def forward(self, x):
        self.mask = (x <= 0)
        out = x.copy()
        out[self.mask] = 0
        return out
    def backward(self, dout):
        dout[self.mask] = 0
        return dout

class Sigmoid(Layer):
    def __init__(self):
        super().__init__()
        self.out = None
    def forward(self, x):
        self.out = 1 / (1 + np.exp(-x))
        return self.out
    def backward(self, dout):
        dx = dout * (1.0 - self.out) * self.out
        return dx
    
def softmax(a):
    c = np.max(a, axis=1, keepdims=True)
    exp_a = np.exp(a - c)
    sum_exp_a = np.sum(exp_a, axis=1, keepdims=True)
    y = exp_a / sum_exp_a
    return y

def cross_entropy_error(y, t):
    if y.ndim == 1:
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)
        
    if t.size == y.size:
        t = t.argmax(axis=1)
             
    batch_size = y.shape[0]
    return -np.sum(np.log(y[np.arange(batch_size), t] + 1e-7)) / batch_size

# losses
class SoftmaxWithLoss:
    def __init__(self):
        self.loss = None    
        self.y = None       
        self.t = None      

    def forward(self, x, t):
        self.t = t
        self.y = softmax(x)
        self.loss = cross_entropy_error(self.y, self.t)
        return self.loss

    def backward(self, dout=1):
        batch_size = self.t.shape[0]
        if self.t.shape == self.y.shape:
            dx = (self.y - self.t) / batch_size
        else:
            dx = self.y.copy()
            labels = self.t
            if labels.ndim != 1:
                labels = labels.reshape(labels.shape[0])
            dx[np.arange(batch_size), labels] -= 1
            dx = dx / batch_size
        return dx
    

class MeanSquaredError:
    def __init__(self):
        self.loss = None
        self.y = None 
        self.t = None 

    def forward(self, y, t):
        self.t = t
        self.y = y
        self.loss = 0.5 * np.sum((y - t)**2) / y.shape[0]
        return self.loss

    def backward(self, dout=1):
        batch_size = self.t.shape[0]
        dx = (self.y - self.t) / batch_size
        return dx