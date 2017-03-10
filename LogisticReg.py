import numpy as np
import math

"""
    This is the class for logistic regression
"""

# compute gradient
def Gradient(x, y, w):
    c = 1 / (1 + math.exp(y * np.dot(w.T, x)))
    g = -y * c * x
    return g

# sigmoid function
def sigmoid(x):
    out = 1 / (1 + math.exp(-x))
    return out

class LogisticRegression:
    """
        gradient descent on the logistic loss: log(1 + exp(-y * w^T * x))
        gradient = -y * x / (1 + exp(-y * w^T * x))
    """
    def __init__(self, targetClass1, targetClass2, weights, learningRate):
        """
            weights: using gradient descent to update the weights
            target: two classes
        """
        self.weights = weights
        self.learningRate = learningRate
        self.targetClass1 = targetClass1
        self.targetClass2 = targetClass2

    def update(self, data):
        x = data.features
        if self.targetClass2 == 0:
            if data.label == self.targetClass1:
                y = 1
            else:
                y = -1

            gradient_loss = Gradient(x, y, self.weights)
            self.weights -= self.learningRate * gradient_loss
        elif data.label == self.targetClass1 or data.label == self.targetClass2:
            if data.label == self.targetClass1:
                y = 1
            elif data.label == self.targetClass2:
                y = -1

            gradient_loss = Gradient(x, y, self.weights)
            self.weights -= self.learningRate * gradient_loss

    def prediction(self, new_data):
        x = new_data.features
        y_predicted = sigmoid(np.dot(self.weights.T, x))
        return y_predicted
