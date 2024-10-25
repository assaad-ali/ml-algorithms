import numpy as np
from ..tree.decision_tree import DecisionTreeClassifier
from sklearn import datasets
from ..utils.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, plot_confusion_matrix

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

    # Initialize the Decision Tree Classifier
    clf = DecisionTreeClassifier(max_depth=3)
    clf.fit(X_train, y_train)

    # Predict on the test set
    y_pred = clf.predict(X_test)

    # Calculate accuracy
    acc = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='macro')
    recall = recall_score(y_test, y_pred, average='macro')
    f1 = f1_score(y_test, y_pred, average='macro')
    conf = plot_confusion_matrix(confusion_matrix(y_test, y_pred), labels=iris.target_names)
    print(f"Accuracy: {acc}\n| Precision: {precision}\n| Recall: {recall}\n| F1 Score: {f1}\n| Confusion Matrix:\n{conf}")

if __name__ == "__main__":
    main()
