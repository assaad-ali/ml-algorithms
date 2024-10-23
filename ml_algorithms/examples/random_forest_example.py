import numpy as np
from ml_algorithms.tree.random_forest import RandomForestClassifier
from sklearn import datasets
from ml_algorithms.utils.metrics import accuracy_score

def main():
    # Load the Iris dataset
    iris = datasets.load_iris()
    X, y = iris.data, iris.target

    # Shuffle the dataset
    indices = np.random.permutation(len(X))
    X = X[indices]
    y = y[indices]

    # Split into training and testing datasets
    split_ratio = 0.8
    split_index = int(len(X) * split_ratio)
    X_train, X_test = X[:split_index], X[split_index:]
    y_train, y_test = y[:split_index], y[split_index:]

    # Initialize the Random Forest Classifier
    clf = RandomForestClassifier(
        n_estimators=10,
        max_depth=5,
        max_features='sqrt',
        bootstrap=True,
        random_state=42
    )

    # Fit the model
    clf.fit(X_train, y_train)

    # Predict on the test set
    y_pred = clf.predict(X_test)

    # Calculate accuracy
    acc = accuracy_score(y_test, y_pred)
    print(f"Random Forest Classifier Accuracy: {acc:.2f}")

if __name__ == "__main__":
    main()
