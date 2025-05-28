import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from ucimlrepo import fetch_ucirepo


def load_and_prepare_data(test_size=0.2, random_state=42):
    # Fetch dataset
    wine_quality = fetch_ucirepo(id=186)

    # Get data
    X = wine_quality.data.features
    y = wine_quality.data.targets['quality']  # Extract target as Series
    # print("Первые 5 строк X:\n", X.head())
    # print("\nПервые 5 значений y:\n", y.head())

    # Handle missing values if any (though dataset says no missing values)
    X = X.fillna(X.mean())

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled, y_train, y_test, scaler

