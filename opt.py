import optuna
from sklearn.model_selection import train_test_split
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
# train_pytorch_model(X_train, y_train, X_test, y_test, optimizer_type='sgd', n_epochs=100,
                        #  scheduler_step=30, optimized_params = [32, 0.01, 0, 0.9]):
        result = train_pytorch_model(
            X_train, y_train,
            X_test, y_test,
            optimizer_type=optimizer_type,
            # learning_rate=learning_rate,
            # weight_decay=weight_decay,
            # momentum=momentum,
            # batch_size=batch_size,
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

#optimized_params = [batch_size, learning_rate, weight_decay, momentum]

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