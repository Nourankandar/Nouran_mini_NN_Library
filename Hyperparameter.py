from neural_network import NeuralNetwork
from optimizers import *
from trainer import Trainer
class Hyperparameter:
    def __init__(self, x_train, t_train, x_test, t_test):
        self.x_train, self.t_train = x_train, t_train
        self.x_test, self.t_test = x_test, t_test

    def search(self, lr_list, batch_sizes, input_size, hidden_list, act_list, output_size):
        results = {}
        for lr in lr_list:
            for b_size in batch_sizes:
                network = NeuralNetwork(input_size=input_size, 
                                        hidden_layers_list=hidden_list, 
                                        activations_list=act_list, 
                                        output_size=output_size)
                
                optimizer = Adam(lr=lr)
                trainer = Trainer(network, self.x_train, self.t_train, self.x_test, self.t_test,
                                  epochs=3, batch_size=b_size, optimizer=optimizer)
                
                _, _, test_acc_list = trainer.fit()
                results[(lr, b_size)] = test_acc_list[-1] 
                
        return results