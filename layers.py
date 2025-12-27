import numpy as np
class Layer:
    def __init__(self):
        pass
    def forward(self, x):
        raise NotImplementedError("forward")
    def backward(self, dout):
        raise NotImplementedError("backward")


class Affine(Layer):
    def __init__(self, W, b):
        super().__init__()
        self.W = W
        self.b = b
        self.x = None
        self.dW = None
        self.db = None

    def forward(self, x):
        self.x = x
        out = np.dot(x, self.W) + self.b  
        return out

    def backward(self, dout):
        dx = np.dot(dout, self.W.T)
        self.dW = np.dot(self.x.T, dout)
        self.db = np.sum(dout, axis=0)
        return dx
    

class Dropout(Layer):
    def __init__(self, dropout_ratio=0.5):
        super().__init__()

        self.dropout_ratio = dropout_ratio
        self.mask = None

    def forward(self, x, train_flg=True):
        if train_flg:
            self.mask = np.random.rand(*x.shape) > self.dropout_ratio
            return x * self.mask
        else:
            return x * (1.0 - self.dropout_ratio)
    def backward(self, dout):
        return dout * self.mask
    

class BatchNormalization(Layer):
    def __init__(self, gamma, beta, momentum=0.9, running_mean=None, running_var=None):
        super().__init__()

        self.gamma = gamma          
        self.beta = beta           
        self.momentum = momentum    
        self.input_shape = None 

        self.running_mean = running_mean
        self.running_var = running_var          
        
        self.batch_size = None
        self.xc = None
        self.xn = None
        self.std = None
        self.dgamma = None
        self.dbeta = None

    def forward(self, x, train_flg=True):
        if self.running_mean is None:
            D = x.shape[1]
            self.running_mean = np.zeros(D)
            self.running_var = np.zeros(D)
                        
        if train_flg:
            mu = x.mean(axis=0)            
            xc = x - mu                   
            var = np.mean(xc**2, axis=0)   
            std = np.sqrt(var + 10e-7)     
            xn = xc / std                 
            
            self.batch_size = x.shape[0]
            self.xc = xc
            self.xn = xn
            self.std = std
            
            self.running_mean = self.momentum * self.running_mean + (1 - self.momentum) * mu
            self.running_var = self.momentum * self.running_var + (1 - self.momentum) * var            
        else:
            xc = x - self.running_mean
            xn = xc / (np.sqrt(self.running_var + 10e-7))
        out = self.gamma * xn + self.beta
        return out

    def backward(self, dout):
        dbeta = dout.sum(axis=0)
        dgamma = np.sum(self.xn * dout, axis=0)
        
        dxn = self.gamma * dout
        dxc = dxn / self.std
        dstd = -np.sum((dxn * self.xc) / (self.std * self.std), axis=0)
        dvar = 0.5 * dstd / self.std
        dxc += (2.0 / self.batch_size) * self.xc * dvar
        dmu = np.sum(dxc, axis=0)
        dx = dxc - dmu / self.batch_size
        
        self.dgamma = dgamma
        self.dbeta = dbeta
        
        return dx