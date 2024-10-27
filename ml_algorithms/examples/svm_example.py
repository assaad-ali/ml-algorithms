import numpy as np
from ml_algorithms.svm.svm import SVM
from sklearn import datasets
from ml_algorithms.utils.metrics import accuracy_score
from ml_algorithms.utils.data_preprocessing import StandardScaler

def main():
    # Load the breast cancer dataset (binary classification)
    bc = datasets.load_breast_cancer()
    X, y = bc.data, bc.target

    # Shuffle the dataset
    indices = np.random.permutation(len(X))
    X = X[indices]
    y = y[indices]

    # Split into training and testing datasets
    split_ratio = 0.8
    split_index = int(len(X) * split_ratio)
    X_train, X_test = X[:split_index], X[split_index:]
    y_train, y_test = y[:split_index], y[split_index:]

    # Standardize features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Initialize the SVM classifier
    svm = SVM(learning_rate=0.001, lambda_param=0.01, n_iters=1000)
    svm.fit(X_train, y_train)

    # Predict on the test set
    y_pred = svm.predict(X_test)
    y_pred = np.where(y_pred == -1, 0, 1)  # Convert back to original labels

    # Calculate accuracy
    acc = accuracy_score(y_test, y_pred)
    print(f"SVM Classifier Accuracy: {acc:.2f}")

if __name__ == "__main__":
    main()
