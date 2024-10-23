
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
