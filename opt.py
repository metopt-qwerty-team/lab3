import optuna
from sklearn.model_selection import train_test_split
from models import train_pytorch_model
from prepare_data import load_and_prepare_data

X_train, X_test, y_train, y_test, _ = load_and_prepare_data()

optimizer_types = ["sgd", "momentum", "nesterov", "adam"]

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
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            momentum=momentum,
            batch_size=batch_size,
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
        print(f"  {key}: {value}")
    print(f"Best MSE: {study.best_value:.4f}")
