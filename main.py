import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from collections import OrderedDict
from neural_network import NeuralNetwork
from Hyperparameter import Hyperparameter
from optimizers import *
from trainer import Trainer
import numpy as np
def main():
    mnist = fetch_openml('mnist_784', version=1, as_frame=False)
    X = mnist.data.astype(np.float32)
    y = mnist.target.astype(int)
    enc = OneHotEncoder(sparse_output=False)
    Y_one_hot = enc.fit_transform(y.reshape(-1, 1))
    x_train, x_test, t_train, t_test = train_test_split(X, Y_one_hot, test_size=0.1, random_state=42)
    
    iters_num = 1000 
    batch_size = 100
    
    print("--- Training with Normalized Data ---")
    x_train_norm = x_train / 255.0
    x_test_norm = x_test / 255.0
    tuner = Hyperparameter(x_train_norm, t_train, x_test_norm, t_test)
    _results = tuner.search(lr_list=[0.01, 0.001], batch_sizes=[64, 128])
    
    best_config = max(_results, key=_results.get)
    best_lr, best_batch = best_config

    
    final_network = NeuralNetwork(input_size=784, 
                                  hidden_layers_list=[100, 50], 
                                  activations_list=['sigmoid', 'relu'], 
                                  output_size=10)

    optimizer = Adam(lr=best_lr)
    trainer = Trainer(final_network, x_train_norm, t_train, x_test_norm, t_test,
                      epochs=5, batch_size=best_batch, optimizer=optimizer)

    loss_list, train_acc_list, test_acc_list = trainer.fit()


    plt.figure(figsize=(10, 4))
    
    # رسم الخسارة
    plt.subplot(1, 2, 1)
    plt.plot(loss_list)
    plt.title("Training Loss")
    plt.xlabel("Iterations")
    
    # رسم الدقة
    plt.subplot(1, 2, 2)
    plt.plot(train_acc_list, label='Train Acc')
    plt.plot(test_acc_list, label='Test Acc')
    plt.title("Accuracy")
    plt.legend()
    
    plt.show()

if __name__ == "__main__":
    main()