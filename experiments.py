import pandas as pd
import time
import psutil
from sklearn.metrics import mean_squared_error
from models import LinearRegressionSGD, train_pytorch_model
from prepare_data import load_and_prepare_data


def batch_size_exp(batch_sizes, torch):
    X_train, X_test, y_train, y_test, _ = load_and_prepare_data()
    if torch:
        batch_sizes.append(len(X_train))
    results = []

    for batch_size in batch_sizes:
        print(f"\nRunning experiment with batch size: {batch_size}")

        t0 = time.perf_counter()

        process = psutil.Process()

        model = LinearRegressionSGD(
            learning_rate=0.01 * batch_size * 10 / len(X_train),
            batch_size=batch_size,
            n_epochs=100
        )
        model.fit(X_train, y_train)

        training_time = time.perf_counter() - t0

        memory_usage = process.memory_info().rss / (1024 * 1024)  # MB

        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)

        results.append({
            'batch_size': batch_size,
            'implementation': 'custom_sgd',
            'mse': mse,
            'training_time': training_time,
            'memory_usage': memory_usage,
            'final_loss': model.loss_history[-1],
            'total_flops': model.total_flops
        })

        if torch:
            bs = batch_size if batch_size != 'full' else len(X_train)
            pt_result = train_pytorch_model(
                X_train, y_train, X_test, y_test,
                optimizer_type='sgd',
                n_epochs=100,
                optimized_params=[bs, 0.01 * bs * 10 / len(X_train), 0, 0.9],
            )
            results.append({
                'batch_size': batch_size,
                'implementation': 'pytorch_sgd',
                'mse': pt_result['final_mse'],
                'training_time': pt_result['training_time'],
                'memory_usage': pt_result['memory_usage'],
                'final_loss': pt_result['test_losses'][-1],
                'total_flops': pt_result['flops']
            })

    return pd.DataFrame(results)


def run_small_batch_size_experiment():
    small_batch_sizes = [1, 16, 32, 64, 128, 256]
    return batch_size_exp(small_batch_sizes, False)


def run_batch_size_experiment():
    batch_sizes = [1, 100, 500, 1000, 1500, 2000, 2500, 3000, 3500, 4000, 4500]
    return batch_size_exp(batch_sizes, True)


# optimized_params = [batch_size, learning_rate, weight_decay, momentum]
optimizer_params = {
    "sgd": [256, 0.004429, 0.000007, 0.0],
    "momentum": [256, 0.000853, 0.000005, 0.783443],
    "nesterov": [64, 0.000547, 0.000741, 0.506646],
    "adagrad": [64, 0.099380, 0.000011, 0.0],
    "rmsprop": [64, 0.006476, 0.000166, 0.0],
    "adam": [256, 0.096550, 0.000385, 0.0]
}


def run_optimizer_comparison():
    X_train, X_test, y_train, y_test, _ = load_and_prepare_data()

    optimizers = ['sgd', 'momentum', 'nesterov', 'adagrad', 'rmsprop', 'adam']
    results = []

    for optimizer in optimizers:
        print(f"\nRunning experiment with optimizer: {optimizer}")

        result = train_pytorch_model(
            X_train, y_train, X_test, y_test,
            optimizer_type=optimizer,
            n_epochs=100,
            optimized_params=optimizer_params[optimizer]
        )

        results.append({
            'optimizer': optimizer,
            'mse': result['final_mse'],
            'training_time': result['training_time'],
            'memory_usage': result['memory_usage'],
            'final_loss': result['test_losses'][-1],
            'train_loss_history': result['train_losses'],
            'test_loss_history': result['test_losses'],
            'total_flops': result['flops'],
        })

    return pd.DataFrame(results)


def run_regularization_experiment():
    X_train, X_test, y_train, y_test, _ = load_and_prepare_data()

    regularizations = [None, 'l1', 'l2', 'elasticnet']
    alphas = [0.0001, 0.001, 0.01, 0.1, 1]
    results = []

    for reg in regularizations:
        for alpha in alphas:
            print(f"\nRunning experiment with regularization: {reg}, alpha: {alpha}")

            start_time = time.time()
            process = psutil.Process()

            model = LinearRegressionSGD(
                learning_rate=0.01,
                batch_size=128,
                n_epochs=100,
                regularization=reg,
                alpha=alpha,
                l1_ratio=0.15 if reg == 'elasticnet' else None
            )
            model.fit(X_train, y_train)

            training_time = time.time() - start_time

            memory_usage = process.memory_info().rss / (1024 * 1024)  # в MB

            y_pred = model.predict(X_test)
            mse = mean_squared_error(y_test, y_pred)

            results.append({
                'regularization': reg,
                'alpha': alpha,
                'mse': mse,
                'training_time': training_time,
                'memory_usage': memory_usage,
                'final_loss': model.loss_history[-1],
                'total_flops': model.total_flops
            })

    return pd.DataFrame(results)


def run_learning_rate_schedule_experiment():
    X_train, X_test, y_train, y_test, _ = load_and_prepare_data()

    schedules = [None, 'time_based', 'step']
    results = []

    for schedule in schedules:
        print(f"\nRunning experiment with learning rate schedule: {schedule}")

        start_time = time.time()
        process = psutil.Process()
        model = LinearRegressionSGD(
            learning_rate=0.001,
            batch_size=64,
            n_epochs=100,
            learning_rate_schedule=schedule
        )
        model.fit(X_train, y_train)

        training_time = time.time() - start_time
        memory_usage = process.memory_info().rss / (1024 * 1024)  # в MB

        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)

        results.append({
            'schedule': "None" if not schedule else schedule,
            'mse': mse,
            'training_time': training_time,
            'memory_usage': memory_usage,
            'final_loss': model.loss_history[-1],
            'loss_history': model.loss_history,
            'total_flops': model.total_flops
        })

    return pd.DataFrame(results)
