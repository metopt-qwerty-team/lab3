import numpy as np
import tracemalloc
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import SGDRegressor
from sklearn.model_selection import cross_val_score
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import time
import resource


class LinearRegressionSGD:
    def __init__(self, learning_rate=0.01, batch_size=32, n_epochs=100,
                 regularization=None, alpha=0.0001, l1_ratio=0.15,
                 learning_rate_schedule=None):
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.regularization = regularization
        self.alpha = alpha
        self.l1_ratio = l1_ratio
        self.learning_rate_schedule = learning_rate_schedule
        self.weights = None
        self.bias = None
        self.loss_history = []
        self.total_flops = 0

    @staticmethod
    def count_flops(batch_size: int, n_features: int) -> dict:
        '''
        exluded:
        1) Regularization operations
        2) Data shuffling operations
        3) Vectorized NumPy operations (count as one operation)
        '''

        B, D = batch_size, n_features
        return {
            'multiply': B * D + D * B + D + D + 1,  # +1 for bias update
            'add': B * (D - 1) + D * (B - 1) + B,  # +B for bias in prediction
            'subtract': B + D + 1,  # +1 for bias update
            'divide': D + 1,  # scaling grad_w и grad_b
            'total': 0
        }

    def _learning_rate(self, epoch):
        if self.learning_rate_schedule == 'time_based':
            return self.learning_rate / (1 + epoch * 0.01)
        elif self.learning_rate_schedule == 'step':
            return self.learning_rate * (0.1 ** (epoch // 30))
        else:
            return self.learning_rate

    def fit(self, X, y):
        y = y.values if hasattr(y, 'values') else y

        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0
        self.total_flops = 0

        flops_per_batch = self.count_flops(min(self.batch_size, n_samples), n_features)
        flops_per_batch['total'] = sum(flops_per_batch.values()) - flops_per_batch['total']

        for epoch in range(self.n_epochs):
            lr = self._learning_rate(epoch)

            indices = np.random.permutation(n_samples)
            X_shuffled = X[indices]
            y_shuffled = y[indices]

            for i in range(0, n_samples, self.batch_size):
                X_batch = X_shuffled[i:i + self.batch_size]
                y_batch = y_shuffled[i:i + self.batch_size]

                y_pred = np.dot(X_batch, self.weights) + self.bias

                error = y_pred - y_batch
                grad_w = (1 / len(X_batch)) * np.dot(X_batch.T, error)
                grad_b = (1 / len(X_batch)) * np.sum(error)

                if self.regularization == 'l2':
                    grad_w += self.alpha * self.weights
                elif self.regularization == 'l1':
                    grad_w += self.alpha * np.sign(self.weights)
                elif self.regularization == 'elasticnet':
                    grad_w += self.alpha * (self.l1_ratio * np.sign(self.weights) +
                                            (1 - self.l1_ratio) * self.weights)

                self.weights -= lr * grad_w
                self.bias -= lr * grad_b
                self.total_flops += flops_per_batch['total']

            y_pred_all = np.dot(X, self.weights) + self.bias
            loss = mean_squared_error(y, y_pred_all)
            self.loss_history.append(loss)

        return self

    def predict(self, X):
        return np.dot(X, self.weights) + self.bias


class PyTorchLinearRegression(nn.Module):
    def __init__(self, input_dim):
        super(PyTorchLinearRegression, self).__init__()
        self.linear = nn.Linear(input_dim, 1)

    def forward(self, x):
        return self.linear(x)


# optimized_params = [batch_size, learning_rate, weight_decay, momentum]
def train_pytorch_model(X_train, y_train, X_test, y_test, optimizer_type='sgd', n_epochs=100,
                        scheduler_step=30, optimized_params=[32, 0.01, 0.0, 0.9]):
    def count_pytorch_flops(model, X_batch):
        B, D = X_batch.shape[0], X_batch.shape[1]
        return {
            'multiply': B * D * 3,  # mul in forward and backward
            'add': B * D * 3,  # add in forward and backward
            'subtract': B,  # error: pred - y
            'divide': 2,  # grad normalization
            'total': B * D * 6 + B + 2  # total approx quan
        }

    batch_size = int(optimized_params[0])
    learning_rate = optimized_params[1]
    weight_decay = optimized_params[2]
    momentum = optimized_params[3]

    X_train_t = torch.FloatTensor(X_train)

    y_train_t = torch.FloatTensor(np.asarray(y_train)).view(-1, 1)
    X_test_t = torch.FloatTensor(X_test)

    y_test_t = torch.FloatTensor(np.asarray(y_test)).view(-1, 1)

    train_dataset = TensorDataset(X_train_t, y_train_t)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    input_dim = X_train.shape[1]
    model = PyTorchLinearRegression(input_dim)
    criterion = nn.MSELoss()

    total_flops = 0
    sample_batch = next(iter(train_loader))[0]
    per_batch_flops = count_pytorch_flops(model, sample_batch)['total']

    if optimizer_type == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    elif optimizer_type == 'momentum':
        optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum,
                              weight_decay=weight_decay)
    elif optimizer_type == 'nesterov':
        optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum,
                              nesterov=True, weight_decay=weight_decay)
    elif optimizer_type == 'adagrad':
        optimizer = optim.Adagrad(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    elif optimizer_type == 'rmsprop':
        optimizer = optim.RMSprop(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    elif optimizer_type == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    else:
        raise ValueError("Unknown optimizer type")

    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=scheduler_step, gamma=0.1)

    train_losses = []
    test_losses = []
    start_time = time.time()

    for epoch in range(n_epochs):
        epoch_losses = []
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            epoch_losses.append(loss.item())

            total_flops += per_batch_flops

        scheduler.step()

        avg_epoch_loss = np.mean(epoch_losses)
        train_losses.append(avg_epoch_loss)

        with torch.no_grad():
            test_outputs = model(X_test_t)
            test_loss = criterion(test_outputs, y_test_t).item()
            test_losses.append(test_loss)

    training_time = time.time() - start_time

    memory_usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 * 1024)  # in MB

    with torch.no_grad():
        y_pred = model(X_test_t).numpy().flatten()
        mse = mean_squared_error(y_test, y_pred)

    return {
        'model': model,
        'train_losses': train_losses,
        'test_losses': test_losses,
        'training_time': training_time,
        'memory_usage': memory_usage,
        'final_mse': mse,
        'flops': total_flops
    }