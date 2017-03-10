import numpy as np

class BayesianLinearRegression:
    """
        Update information vector J and precision matrix P given new data point, then use the final mean and
        varaince to do prediction. The predicted label is the mean of prediction.
    """

    def __init__(self, targetClass1, targetClass2, mean, covariance, sigma=0):
        """
            sigma: variance of the output noise.
            p_mean: mean vector of the prior p(theta)
            p_covariance: covariance matrix of the prior p(theta)
            mean: final mean
            covariance: final covariance
            J: information vector
            P: precision matrix
        """
        # attributes are not changing during Update
        self.targetClass1 = targetClass1
        self.targetClass2 = targetClass2
        self.sigma2 = sigma ** 2
        self.p_mean = mean
        self.p_covariance = covariance

        # updating attributes
        self.mean = mean
        self.covariance = covariance
        self.P = np.linalg.inv(covariance)
        self.J = np.dot(self.P, mean[:, None])

    def updateRule(self, data):
        # get current x and y
        x = data.features
        if self.targetClass2 == 0:
            if data.label == self.targetClass1:
                y = 1
            else:
                y = -1

            # update J and P
            self.J += (y * x[:, None]) / float(self.sigma2)
            self.P += (1 / float(self.sigma2)) * np.dot(x[:, None], x[None, :])
        elif data.label == self.targetClass1 or data.label == self.targetClass2:
            # print("Find a traget class.")
            if data.label == self.targetClass1:
                y = 1
            elif data.label == self.targetClass2:
                y = -1

            # update J and P
            self.J += (y * x[:, None]) / float(self.sigma2)
            self.P += (1 / float(self.sigma2)) * np.dot(x[:, None], x[None, :])

    def set_final_parameters(self):
        self.covariance = np.linalg.inv(self.P)
        self.mean = np.dot(self.covariance, self.J)

    def prediction(self, new_data):
        x = new_data.features
        predicted_mean = np.dot(self.mean.T, x)
        predicted_covariance = np.dot(np.dot(x[None, :], self.covariance), x[:, None]) + self.sigma2
        return predicted_mean, predicted_covariance
