
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
