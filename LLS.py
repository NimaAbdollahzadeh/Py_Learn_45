import numpy as np
from numpy.linalg import inv

class LLS:
    def __init__(self):
        self.w = None

    def fit(self, X_train, Y_train):
        self.w = inv(X_train.T @ X_train) @ X_train.T @ Y_train
        return self.w
    
    def predict(self, X_test):
        Y_pred = X_test @ self.w
        return Y_pred
    
    def evaluate(self, X_test, Y_test, loss='MAE'):
        Y_pred = self.predict(X_test)
        Error = Y_test - Y_pred

        if loss == 'MAE':
            return np.mean(np.abs(Error))
        elif loss == 'MSE':
            return np.mean(Error * 2)
        elif loss == 'RMSE':
            return np.sqrt(np.mean(Error ** 2))