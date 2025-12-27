import numpy as np
from layers import Affine
from activationsLosses import *
import numpy as np
from layers import Affine, BatchNormalization, Dropout
from activationsLosses import *

class NeuralNetwork:
    def __init__(self, input_size, hidden_layers_list, activations_list, output_size, weight_init_std=0.01):
        self.params = {}
        self.layers = []
        
        all_sizes = [input_size] + hidden_layers_list + [output_size]

        for i in range(len(hidden_layers_list)):
            act_type = activations_list[i].lower()
            idx = i + 1 
            n_in = all_sizes[i]
            n_out = all_sizes[i+1]
            if act_type == 'relu':
                # He Initialization 
                scale = np.sqrt(2.0 / n_in)
            elif act_type in ['sigmoid', 'tanh']:
                # Xavier Initialization
                scale = np.sqrt(1.0 / n_in)
            else:
                scale = 0.01

            self.params[f'W{idx}'] = scale * np.random.randn(n_in, n_out)
            self.params[f'b{idx}'] = np.zeros(n_out)
            self.layers.append(Affine(self.params[f'W{idx}'], self.params[f'b{idx}']))
            
            self.params[f'gamma{idx}'] = np.ones(n_out)
            self.params[f'beta{idx}'] = np.zeros(n_out)
            self.layers.append(BatchNormalization(self.params[f'gamma{idx}'], self.params[f'beta{idx}']))

            
            if act_type == 'relu':
                self.layers.append(Relu())
            elif act_type == 'sigmoid':
                self.layers.append(Sigmoid())
            elif act_type == 'tanh':
                self.layers.append(Tanh())


        last_idx = len(hidden_layers_list) + 1
        scale_out = np.sqrt(1.0 / all_sizes[-2])
        self.params[f'W{last_idx}'] = scale_out * np.random.randn(all_sizes[-2], all_sizes[-1])
        self.params[f'b{last_idx}'] = np.zeros(all_sizes[-1])
        self.layers.append(Affine(self.params[f'W{last_idx}'], self.params[f'b{last_idx}']))
        
        self.lastLayer = SoftmaxWithLoss()

    def predict(self, x, train_flg=False):
        for layer in self.layers:
            if isinstance(layer, (Dropout, BatchNormalization)):
                x = layer.forward(x, train_flg)
            else:
                x = layer.forward(x)
        return x
        
    def gradient(self, x, t):
        self.loss(x, t, train_flg=True)

        dout = 1  
        dout = self.lastLayer.backward(dout)

        layers = list(self.layers)
        layers.reverse()
        for layer in layers:
            dout = layer.backward(dout)

        grads = {}
        affine_idx = 1
        bn_idx = 1
        
        for layer in self.layers:
            if isinstance(layer, Affine):
                grads[f'W{affine_idx}'] = layer.dW
                grads[f'b{affine_idx}'] = layer.db
                affine_idx += 1
            elif isinstance(layer, BatchNormalization):
                grads[f'gamma{bn_idx}'] = layer.dgamma
                grads[f'beta{bn_idx}'] = layer.dbeta
                bn_idx += 1
                
        return grads
    def loss(self, x, t, train_flg=False):
        y = self.predict(x, train_flg)
        return self.lastLayer.forward(y, t)

    def accuracy(self, x, t):
        y = self.predict(x, train_flg=False)
        y = np.argmax(y, axis=1) 
        if t.ndim != 1: t = np.argmax(t, axis=1) 
        
        return np.sum(y == t) / float(x.shape[0])
    
    def summary(self):
        print("\n--- Network Summary ---")
        for i, layer in enumerate(self.layers):
            name = layer.__class__.__name__
            if name == "Affine":
                print(f"Layer {i+1} [Affine]: W shape {layer.W.shape}, b shape {layer.b.shape}")
            else:
                print(f"Layer {i+1} [{name}]: Activation function")
        print("-----------------------\n")
