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
            'multiply': B*D + D*B + D + D + 1,  # +1 for bias update
            'add': B*(D-1) + D*(B-1) + B,       # +B for bias in prediction
            'subtract': B + D + 1,              # +1 for bias update
            'divide': D + 1,                    # scaling grad_w и grad_b
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
        # Convert y to numpy array if it's a pandas Series
        y = y.values if hasattr(y, 'values') else y

        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0
        self.total_flops = 0

        flops_per_batch = self.count_flops(min(self.batch_size, n_samples), n_features)
        flops_per_batch['total'] = sum(flops_per_batch.values()) - flops_per_batch['total']

        for epoch in range(self.n_epochs):
            lr = self._learning_rate(epoch)

            # Shuffle data
            indices = np.random.permutation(n_samples)
            X_shuffled = X[indices]
            y_shuffled = y[indices]

            for i in range(0, n_samples, self.batch_size):
                X_batch = X_shuffled[i:i + self.batch_size]
                y_batch = y_shuffled[i:i + self.batch_size]

                # Predictions
                y_pred = np.dot(X_batch, self.weights) + self.bias

                # Compute gradients
                error = y_pred - y_batch
                grad_w = (1 / len(X_batch)) * np.dot(X_batch.T, error)
                grad_b = (1 / len(X_batch)) * np.sum(error)

                # Add regularization
                if self.regularization == 'l2':
                    grad_w += self.alpha * self.weights
                elif self.regularization == 'l1':
                    grad_w += self.alpha * np.sign(self.weights)
                elif self.regularization == 'elasticnet':
                    grad_w += self.alpha * (self.l1_ratio * np.sign(self.weights) +
                                            (1 - self.l1_ratio) * self.weights)

                # Update parameters
                self.weights -= lr * grad_w
                self.bias -= lr * grad_b
                self.total_flops += flops_per_batch['total']


            # Compute and store loss for monitoring
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

#optimized_params = [batch_size, learning_rate, weight_decay, momentum]
def train_pytorch_model(X_train, y_train, X_test, y_test, optimizer_type='sgd', n_epochs=100,
                         scheduler_step=30, optimized_params = [32, 0.01, 0.0, 0.9]):
    
    def count_pytorch_flops(model, X_batch):
        B, D = X_batch.shape[0], X_batch.shape[1]
        # Для линейного слоя: D умножений и D сложений на пример
        # Плюс аналогичные операции для backward pass (примерно в 2-3 раза больше)
        return {
            'multiply': B*D * 3,  # mul in forward and backward
            'add': B*D * 3,        # add in forward and backward
            'subtract': B,          # error: pred - y
            'divide': 2,            # grad normalization
            'total': B*D*6 + B + 2  # total approx quan
        }

    total_flops = 0

    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # if torch.cuda.is_available():
    #     torch.cuda.reset_peak_memory_stats()
    batch_size = int(optimized_params[0])
    learning_rate = optimized_params[1]
    weight_decay = optimized_params[2]
    momentum = optimized_params[3]

    # Convert to tensors
    X_train_t = torch.FloatTensor(X_train)
    # y_train_t = torch.FloatTensor(y_train.values).view(-1, 1)
    y_train_t = torch.FloatTensor(np.asarray(y_train)).view(-1, 1)
    X_test_t = torch.FloatTensor(X_test)
    # y_test_t = torch.FloatTensor(y_test.values).view(-1, 1)
    y_test_t = torch.FloatTensor(np.asarray(y_test)).view(-1, 1)

    # Create dataset and dataloader
    train_dataset = TensorDataset(X_train_t, y_train_t)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # Model
    input_dim = X_train.shape[1]
    model = PyTorchLinearRegression(input_dim)
    criterion = nn.MSELoss()

    # Optimizer
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

    # Learning rate scheduler
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=scheduler_step, gamma=0.1)

    # Training
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

            batch_flops = count_pytorch_flops(model, X_batch)
            total_flops += batch_flops['total']

        scheduler.step()

        # Compute average epoch loss
        avg_epoch_loss = np.mean(epoch_losses)
        train_losses.append(avg_epoch_loss)

        # Compute test loss
        with torch.no_grad():
            test_outputs = model(X_test_t)
            test_loss = criterion(test_outputs, y_test_t).item()
            test_losses.append(test_loss)

    training_time = time.time() - start_time
    # if torch.cuda.is_available():
    #     memory_usage = torch.cuda.max_memory_allocated() / (1024 * 1024)  # in MB
    # else:
    #     memory_usage = -1
    memory_usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 * 1024)  # in MB

    # Final evaluation
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
