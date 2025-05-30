import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_regression
from sklearn.preprocessing import StandardScaler

X, y = make_regression(
    n_samples=1000,
    n_features=10,
    noise=20.0,
    random_state=42
)

X = StandardScaler().fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

gbr = GradientBoostingRegressor(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=3,
    random_state=42
)
gbr.fit(X_train, y_train)


train_mse = []
test_mse  = []
for y_pred_train in gbr.staged_predict(X_train):
    train_mse.append(mean_squared_error(y_train, y_pred_train))
for y_pred_test in gbr.staged_predict(X_test):
    test_mse.append(mean_squared_error(y_test, y_pred_test))


plt.figure(figsize=(10, 6))
iterations = np.arange(1, len(train_mse) + 1)
plt.plot(iterations, train_mse, label='Train MSE', linewidth=2)
plt.plot(iterations, test_mse,  label='Test MSE',  linewidth=2)
plt.xlabel('Boosting Iteration', fontsize=12)
plt.ylabel('Mean Squared Error', fontsize=12)
plt.title('Gradient Boosting: Train vs Test MSE Convergence', fontsize=14)
plt.legend(fontsize=12)
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig('gradient_boosting_convergence.png')