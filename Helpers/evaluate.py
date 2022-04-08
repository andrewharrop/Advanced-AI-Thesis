
   
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
from tensorflow.keras.models import Model


def evalute_model(model: Model, test_X: list, test_Y: list, batch_size: int = 8) -> tuple:
    """
        Evaluate model perfromance
        
        :param model: The CNN model to evaluate.
        :param test_X: The test images.
        :param test_Y: The test labels.
        :param batch_size: The batch size.
        :return: The accuracy and the confusion matrix.
    """

    # Predict the test set
    y_pred = model.predict(test_X, batch_size=batch_size)
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_true = np.argmax(test_Y, axis=1)

    print(f"Actuals: {y_true}")
    print(f"Predictions: {y_pred_classes}")

    # Print the confusion matrix
    print(confusion_matrix(y_true, y_pred_classes))

    # Print the classification report
    print(classification_report(y_true, y_pred_classes, target_names=['no', 'yes']))

    total = sum(sum(confusion_matrix(y_true, y_pred_classes)))
    accuracy = (confusion_matrix(y_true, y_pred_classes)[0][0] + confusion_matrix(y_true, y_pred_classes)[1][1]) / total
    print("Accuracy: %f" % accuracy)

    return accuracy, confusion_matrix(y_true, y_pred_classes)