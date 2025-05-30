import optuna
import numpy as np
from sklearn.model_selection import train_test_split
from models import LinearRegressionSGD
from models import mean_squared_error
from models import train_pytorch_model
from prepare_data import load_and_prepare_data

optuna.logging.set_verbosity(optuna.logging.WARNING)

X_train, X_test, y_train, y_test, _ = load_and_prepare_data()

optimizer_types = ["sgd", "momentum", "nesterov", "adagrad", "rmsprop", "adam"]

best_results = {}


def make_objective(optimizer_type):
    def objective(trial):
        learning_rate = trial.suggest_float("learning_rate", 1e-4, 1e-1, log=True)
        weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True)
        batch_size = trial.suggest_categorical("batch_size", [16, 32, 64, 128, 256])

        momentum = 0.0
        if optimizer_type in ["momentum", "nesterov"]:
            momentum = trial.suggest_float("momentum", 0.5, 0.99)

        result = train_pytorch_model(
            X_train, y_train,
            X_test, y_test,
            optimizer_type=optimizer_type,
            optimized_params=[batch_size, learning_rate, weight_decay, momentum],
            n_epochs=50
        )

        return result["final_mse"]

    return objective


for optimizer in optimizer_types:
    print(f"\n Optimizing: {optimizer.upper()}")

    study = optuna.create_study(direction="minimize")
    study.optimize(make_objective(optimizer), n_trials=30)

    best_results[optimizer] = {
        "best_params": study.best_params,
        "best_value": study.best_value
    }

    print("Best params:")
    for key, value in study.best_params.items():
        print(f"  {key}: {value:.6f}")
    print(f"Best MSE: {study.best_value:.4f}")


def make_objective_sgd():
    def objective(trial):
        lr = 0.01
        # lr = trial.suggest_float("learning_rate", 1e-5, 0.5, log=True)
        batch_size = trial.suggest_categorical("batch_size", [16, 32, 64, 128, 256, 512, 1024])
        n_epochs = trial.suggest_categorical("n_epochs", [50, 100, 150])

        alpha = trial.suggest_float("alpha", 1e-6, 1e-2, log=True)
        l1_ratio = trial.suggest_float("l1_ratio", 0, 1) if alpha > 1e-5 else 0

        model = LinearRegressionSGD(
            learning_rate=lr,
            batch_size=batch_size,
            n_epochs=n_epochs,
            regularization='elasticnet' if alpha > 1e-5 else None,
            alpha=alpha,
            l1_ratio=l1_ratio
        )

        mse_values = []
        for _ in range(3):
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            mse_values.append(mean_squared_error(y_test, y_pred))

        return np.mean(mse_values)

    return objective


study = optuna.create_study(direction="minimize")
study.optimize(make_objective_sgd(), n_trials=50)

print("Best params:")
for key, value in study.best_params.items():
    print(f"  {key}: {value:.6f}")
print(f"Best MSE: {study.best_value:.4f}")

'''
Best params:
  learning_rate: 0.000748
  batch_size: 64.000000
  n_epochs: 100.000000
  alpha: 0.009475
  l1_ratio: 0.401440
Best MSE: 0.5432



learning_rate: 0.000254
  batch_size: 16.000000
  n_epochs: 100.000000
  alpha: 0.009916
  l1_ratio: 0.970680
Best MSE: 0.5422
'''

# optimized_params = [batch_size, learning_rate, weight_decay, momentum]

'''
Optimizing: SGD
Best params:
  learning_rate: 0.004429
  weight_decay: 0.000007
  batch_size: 256.000000
Best MSE: 0.5441

 Optimizing: MOMENTUM
Best params:
  learning_rate: 0.000853
  weight_decay: 0.000005
  batch_size: 256.000000
  momentum: 0.783443
Best MSE: 0.5447

 Optimizing: NESTEROV
Best params:
  learning_rate: 0.000547
  weight_decay: 0.000741
  batch_size: 64.000000
  momentum: 0.506646
Best MSE: 0.5452

 Optimizing: ADAGRAD
Best params:
  learning_rate: 0.099380
  weight_decay: 0.000011
  batch_size: 64.000000
Best MSE: 0.5786

 Optimizing: RMSPROP
Best params:
  learning_rate: 0.006476
  weight_decay: 0.000166
  batch_size: 64.000000
Best MSE: 0.5466

 Optimizing: ADAM
Best params:
  learning_rate: 0.096550
  weight_decay: 0.000385
  batch_size: 256.000000
Best MSE: 0.5456
'''