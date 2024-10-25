import numpy as np

def accuracy_score(y_true, y_pred):
    """
    Calculate the accuracy between true and predicted labels.

    Parameters:
        y_true (ndarray): True labels.
        y_pred (ndarray): Predicted labels.

    Returns:
        float: Accuracy score.
    """
    correct = sum(y_t == y_p for y_t, y_p in zip(y_true, y_pred))
    return correct / len(y_true)

def precision_score(y_true, y_pred, average='binary'):
    """
    Calculates the precision of the predictions.
    
    Parameters:
        y_true (ndarray): True labels.
        y_pred (ndarray): Predicted labels.
        average (str): Type of averaging performed ('binary', 'macro', 'micro', 'weighted').
    
    Returns:
        float: Precision score.
    """
    if average == 'binary':
        # Binary classification: Precision = TP / (TP + FP)
        tp = np.sum((y_true == 1) & (y_pred == 1))
        fp = np.sum((y_true == 0) & (y_pred == 1))
        return tp / (tp + fp) if (tp + fp) > 0 else 0.0

    elif average in ['macro', 'micro', 'weighted']:
        labels = np.unique(y_true)
        precisions = []
        for label in labels:
            tp = np.sum((y_true == label) & (y_pred == label))
            fp = np.sum((y_true != label) & (y_pred == label))
            precisions.append(tp / (tp + fp) if (tp + fp) > 0 else 0.0)
        if average == 'macro':
            return np.mean(precisions)
        elif average == 'micro':
            tp = np.sum(y_true == y_pred)
            fp = np.sum((y_true != y_pred) & (y_pred != None))
            return tp / (tp + fp) if (tp + fp) > 0 else 0.0
        elif average == 'weighted':
            weights = [np.sum(y_true == label) for label in labels]
            return np.average(precisions, weights=weights)
    else:
        raise ValueError("Unsupported 'average' argument. Choose from 'binary', 'macro', 'micro', 'weighted'.")

def recall_score(y_true, y_pred, average='binary'):
    """
    Calculates the recall of the predictions.
    
    Parameters:
        y_true (ndarray): True labels.
        y_pred (ndarray): Predicted labels.
        average (str): Type of averaging performed ('binary', 'macro', 'micro', 'weighted').
    
    Returns:
        float: Recall score.
    """
    if average == 'binary':
        # Binary classification: Recall = TP / (TP + FN)
        tp = np.sum((y_true == 1) & (y_pred == 1))
        fn = np.sum((y_true == 1) & (y_pred == 0))
        return tp / (tp + fn) if (tp + fn) > 0 else 0.0

    elif average in ['macro', 'micro', 'weighted']:
        labels = np.unique(y_true)
        recalls = []
        for label in labels:
            tp = np.sum((y_true == label) & (y_pred == label))
            fn = np.sum((y_true == label) & (y_pred != label))
            recalls.append(tp / (tp + fn) if (tp + fn) > 0 else 0.0)
        if average == 'macro':
            return np.mean(recalls)
        elif average == 'micro':
            tp = np.sum(y_true == y_pred)
            fn = np.sum((y_true != y_pred) & (y_true != None))
            return tp / (tp + fn) if (tp + fn) > 0 else 0.0
        elif average == 'weighted':
            weights = [np.sum(y_true == label) for label in labels]
            return np.average(recalls, weights=weights)
    else:
        raise ValueError("Unsupported 'average' argument. Choose from 'binary', 'macro', 'micro', 'weighted'.")

def f1_score(y_true, y_pred, average='binary'):
    """
    Calculates the F1 score.
    
    Parameters:
        y_true (ndarray): True labels.
        y_pred (ndarray): Predicted labels.
        average (str): Type of averaging performed ('binary', 'macro', 'micro', 'weighted').
    
    Returns:
        float: F1 score.
    """
    prec = precision_score(y_true, y_pred, average=average)
    rec = recall_score(y_true, y_pred, average=average)
    return 2 * (prec * rec) / (prec + rec) if (prec + rec) != 0 else 0

import numpy as np

def confusion_matrix(y_true, y_pred, labels=None):
    """
    Computes the confusion matrix with labeled rows and columns.
    
    Parameters:
        y_true (ndarray): True labels.
        y_pred (ndarray): Predicted labels.
        labels (list): List of labels to index the matrix.
    
    Returns:
        tuple: (Confusion matrix, Labels)
            - Confusion matrix (ndarray): The confusion matrix with shape (n_classes, n_classes)
            - Labels (ndarray): The labels corresponding to the rows and columns.
    """
    if labels is None:
        labels = np.unique(np.concatenate((y_true, y_pred)))
        
    matrix = np.zeros((len(labels), len(labels)), dtype=int)
    label_to_index = {label: index for index, label in enumerate(labels)}
    
    for t, p in zip(y_true, y_pred):
        i = label_to_index[t]
        j = label_to_index[p]
        matrix[i, j] += 1
    
    return matrix

def plot_confusion_matrix(matrix, labels):
    """
    Prints the confusion matrix with labels for better understanding.
    
    Parameters:
        matrix (ndarray): Confusion matrix.
        labels (list or ndarray): List or array of labels corresponding to the matrix rows and columns.
    """
    result = f"Predicted labels -> {labels}\n"
    
    for i, row in enumerate(matrix):
        result += f"Actual label {labels[i]}: {row}\n"
    
    return result
