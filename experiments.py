import pandas as pd
import time
import psutil
from sklearn.metrics import mean_squared_error
from models import LinearRegressionSGD, train_pytorch_model
from prepare_data import load_and_prepare_data


def run_batch_size_experiment():
    X_train, X_test, y_train, y_test, _ = load_and_prepare_data()

    batch_sizes = [1, 100, 500, 1000, 1500, 2000, 2500, 3000, 3500, 4000, 4500, len(X_train)]
    results = []

    for batch_size in batch_sizes:
        print(f"\nRunning experiment with batch size: {batch_size}")

        # Custom SGD implementation
        start_time = time.time()
        # mem_before = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss

        process = psutil.Process()

        model = LinearRegressionSGD(
            learning_rate=0.01,
            batch_size=batch_size,
            n_epochs=100
        )
        model.fit(X_train, y_train)

        # mem_after = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        training_time = time.time() - start_time
        # memory_usage = (mem_after - mem_before) / (1024 * 1024)  # in MB
        memory_usage = process.memory_info().rss / (1024 * 1024)  # в MB

        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)

        results.append({
            'batch_size': batch_size,
            'implementation': 'custom_sgd',
            'mse': mse,
            'training_time': training_time,
            'memory_usage': memory_usage,
            'final_loss': model.loss_history[-1]
        })

        # PyTorch SGD
        pt_result = train_pytorch_model(
            X_train, y_train, X_test, y_test,
            optimizer_type='sgd',
            batch_size=batch_size if batch_size != 'full' else len(X_train),
            learning_rate=0.01,
            n_epochs=100
        )

        results.append({
            'batch_size': batch_size,
            'implementation': 'pytorch_sgd',
            'mse': pt_result['final_mse'],
            'training_time': pt_result['training_time'],
            'memory_usage': pt_result['memory_usage'],
            'final_loss': pt_result['test_losses'][-1]
        })

    return pd.DataFrame(results)


def run_optimizer_comparison():
    X_train, X_test, y_train, y_test, _ = load_and_prepare_data()

    optimizers = ['sgd', 'momentum', 'nesterov', 'adagrad', 'rmsprop', 'adam']
    results = []

    for optimizer in optimizers:
        print(f"\nRunning experiment with optimizer: {optimizer}")

        result = train_pytorch_model(
            X_train, y_train, X_test, y_test,
            optimizer_type=optimizer,
            batch_size=500,  # ???
            learning_rate=0.01,
            n_epochs=100
        )

        results.append({
            'optimizer': optimizer,
            'mse': result['final_mse'],
            'training_time': result['training_time'],
            'memory_usage': result['memory_usage'],
            'final_loss': result['test_losses'][-1],
            'train_loss_history': result['train_losses'],
            'test_loss_history': result['test_losses']
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
            # mem_before = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            process = psutil.Process()

            model = LinearRegressionSGD(
                learning_rate=0.01,
                batch_size=500,  # ???,
                n_epochs=100,
                regularization=reg,
                alpha=alpha,
                l1_ratio=0.15 if reg == 'elasticnet' else None
            )
            model.fit(X_train, y_train)

            # mem_after = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
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
                'final_loss': model.loss_history[-1]
            })

    return pd.DataFrame(results)


def run_learning_rate_schedule_experiment():
    X_train, X_test, y_train, y_test, _ = load_and_prepare_data()

    schedules = [None, 'time_based', 'step']
    results = []

    for schedule in schedules:
        print(f"\nRunning experiment with learning rate schedule: {schedule}")

        start_time = time.time()
        # mem_before = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        process = psutil.Process()

        model = LinearRegressionSGD(
            learning_rate=0.1,  # Higher initial rate for schedules
            batch_size=500,  # ???
            n_epochs=100,
            learning_rate_schedule=schedule
        )
        model.fit(X_train, y_train)

        # mem_after = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        training_time = time.time() - start_time
        # memory_usage = (mem_after - mem_before) / (1024 * 1024)  # in MB
        memory_usage = process.memory_info().rss / (1024 * 1024)  # в MB

        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)

        results.append({
            'schedule': schedule,
            'mse': mse,
            'training_time': training_time,
            'memory_usage': memory_usage,
            'final_loss': model.loss_history[-1],
            'loss_history': model.loss_history
        })

    return pd.DataFrame(results)
