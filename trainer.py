import numpy as np

class Trainer:
    def __init__(self, network, x_train, t_train, x_test, t_test,
                 epochs=20, batch_size=100, optimizer=None):
        self.network = network
        self.x_train = x_train
        self.t_train = t_train
        self.x_test = x_test
        self.t_test = t_test
        self.epochs = epochs
        self.batch_size = batch_size
        self.optimizer = optimizer
        
        self.train_loss_list = []
        self.train_acc_list = []
        self.test_acc_list = []

    def train_step(self):
        train_size = self.x_train.shape[0]
        batch_mask = np.random.choice(train_size, self.batch_size)
        x_batch = self.x_train[batch_mask]
        t_batch = self.t_train[batch_mask]
        grads = self.network.gradient(x_batch, t_batch)
        self.optimizer.update(self.network.params, grads)
        loss = self.network.loss(x_batch, t_batch, train_flg=True)
        return loss

    def fit(self):
        train_size = self.x_train.shape[0]
        iter_per_epoch = max(train_size / self.batch_size, 1)
        max_iter = int(self.epochs * iter_per_epoch)

        for i in range(max_iter):
            loss = self.train_step()
            self.train_loss_list.append(loss)

            if i % int(iter_per_epoch) == 0:
                epoch_num = int(i / iter_per_epoch)
                train_acc = self.network.accuracy(self.x_train, self.t_train)
                test_acc = self.network.accuracy(self.x_test, self.t_test)
                
                self.train_acc_list.append(train_acc)
                self.test_acc_list.append(test_acc)
                print(f"Epoch {epoch_num} | Loss: {loss:.4f} | Train Acc: {train_acc:.4f} | Test Acc: {test_acc:.4f}")
        
        return self.train_loss_list, self.train_acc_list, self.test_acc_list