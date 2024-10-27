import numpy as np
from ml_algorithms.tree.decision_tree_regressor import DecisionTreeRegressor
from sklearn import datasets
from ..utils.metrics import mean_squared_error, mean_absolute_error, r2_score

def main():
    # Load the California Housing dataset
    california = datasets.fetch_california_housing()
    X, y = california.data, california.target

    # Shuffle and split the dataset
    indices = np.random.permutation(len(X))
    X = X[indices]
    y = y[indices]
    split_ratio = 0.8
    split_index = int(len(X) * split_ratio)
    X_train, X_test = X[:split_index], X[split_index:]
    y_train, y_test = y[:split_index], y[split_index:]

    # Initialize the Decision Tree Regressor
    reg = DecisionTreeRegressor(max_depth=3)
    reg.fit(X_train, y_train)

    # Predict on the test set
    y_pred = reg.predict(X_test)

    # Calculate regression metrics
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(f"Mean Squared Error: {mse:.2f}")
    print(f"Mean Absolute Error: {mae:.2f}")
    print(f"R-squared: {r2:.2f}")



if __name__ == "__main__":
    main()
