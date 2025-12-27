from neural_network import NeuralNetwork
from optimizers import *
from trainer import Trainer
class Hyperparameter:
    def __init__(self, x_train, t_train, x_test, t_test):
        self.x_train, self.t_train = x_train, t_train
        self.x_test, self.t_test = x_test, t_test

    def search(self, lr_list, batch_sizes):
        
        results = {}
        for lr in lr_list:
            for b_size in batch_sizes:
                print(f"Testing with lr: {lr}, batch_size: {b_size}")
                
                # بناء الشبكة المطلوبة في التوصيف للاختبار
                # Dense > Sigmoid > BatchNorm > Dense > Relu > Dense > SoftmaxWithLoss
                network = NeuralNetwork(input_size=784, 
                                        hidden_layers_list=[100, 50], 
                                        activations_list=['sigmoid', 'relu'], 
                                        output_size=10)
                
                optimizer = Adam(lr=lr)
                trainer = Trainer(network, self.x_train, self.t_train, self.x_test, self.t_test,
                                  epochs=3, batch_size=b_size, optimizer=optimizer)
                
                _, _, test_acc_list = trainer.fit()
                results[(lr, b_size)] = test_acc_list[-1] 
                
        return results