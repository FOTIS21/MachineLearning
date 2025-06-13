import numpy as np
from skimage.metrics import mean_squared_error
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from linear_regression import LinearRegression

# Φόρτωση του dataset
data = fetch_california_housing()
X = data.data
y = data.target

rmse_list = []

# Διαχωρισμός σε train/test sets (70% / 30%)
for i in range(20):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=i
    )
    model = LinearRegression()
    model.fit(X_train, y_train)

    _, test_mse = model.evaluate(X_test, y_test)
    test_rmse = np.sqrt(test_mse)
    rmse_list.append(test_mse)

mean_rmse = np.mean(rmse_list)
std_rmse = np.std(rmse_list)

print(f"Μέση τιμή RMSE: {mean_rmse:.4f}")
print(f"Τυπική απόκλιση RMSE: {std_rmse:.4f}")

# 20 επαναλήψεις
for i in range(20):
    print(f" {i}")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=i
    )

    model = LinearRegression()
    model.fit(X_train, y_train)

    y_test_pred = model.predict(X_test)
    test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
    rmse_list.append(test_rmse)

# Υπολογισμός μέσης τιμής και τυπικής απόκλισης
mean_rmse = np.mean(rmse_list)
std_rmse = np.std(rmse_list)

print(f"[scikit-learn] Μέση τιμή RMSE: {mean_rmse:.4f}")
print(f"[scikit-learn] Τυπική απόκλιση RMSE: {std_rmse:.4f}")