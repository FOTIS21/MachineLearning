import numpy as np

class LinearRegression:
    def __init__(self):
        self.w = None
        self.b = None

    def fit(self, X, y):
        if not isinstance(X, np.ndarray) or not isinstance(y, np.ndarray):
            raise ValueError("X και y πρέπει να είναι numpy arrays.")

        if X.shape[0] != y.shape[0]:
            raise ValueError("Το πλήθος γραμμών του X και του y πρέπει να είναι ίδιο.")

        # Δημιουργία πίνακα σχεδιασμού με πρόσθετη στήλη 1s
        ones = np.ones((X.shape[0], 1))
        X_aug = np.hstack((X, ones))

        # Κανονικές εξισώσεις: θ = (XᵀX)^(-1) Xᵀy
        theta = np.dot(np.linalg.inv(np.dot(X_aug.T, X_aug)), np.dot(X_aug.T, y))

        self.w = theta[:-1]
        self.b = theta[-1]

    def predict(self, X):
        if self.w is None or self.b is None:
            raise ValueError("Το μοντέλο δεν έχει εκπαιδευτεί.")

        return np.dot(X, self.w) + self.b

    def evaluate(self, X, y):
        if self.w is None or self.b is None:
            raise ValueError("Το μοντέλο δεν έχει εκπαιδευτεί.")

        y_pred = self.predict(X)
        mse = (1 / X.shape[0]) * np.dot((y_pred - y).T, (y_pred - y))
        return y_pred, mse
