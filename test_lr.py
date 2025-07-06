import numpy as np
from sklearn.linear_model import LinearRegression as SKLearn
from sklearn.metrics import mean_squared_error
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from linear_regression import LinearRegression

# Φόρτωση του dataset
data = fetch_california_housing()
X = data.data
y = data.target

rmse_list = []

# Διαχωρισμός σε train/test sets (70% / 30%)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)
model = LinearRegression()
model.fit(X_train, y_train)

_, test_mse = model.evaluate(X_test, y_test)
test_rmse = np.sqrt(test_mse)

print(f"RMSE: {test_rmse:.4f}")

sk_learn_model = SKLearn()
sk_learn_model.fit(X_train, y_train)
y_pred = sk_learn_model.predict(X_test)
sk_mse = mean_squared_error(y_test, y_pred)
sk_rmse = np.sqrt(sk_mse)
print(f"\n *[scikit-learn]* \nRMSE: {sk_rmse:.4f}")

sk_rmse_list = []

# 20 επαναλήψεις
for i in range(20):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3
    )
    model = LinearRegression()
    model.fit(X_train, y_train)
    _, test_mse = model.evaluate(X_test, y_test)
    rmse_list.append(np.sqrt(test_mse))

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
    sk_linear_model = SKLearn()
    sk_linear_model.fit(X_train, y_train)
    y_pred = sk_linear_model.predict(X_test)
    sk_mse = mean_squared_error(y_test, y_pred)
    sk_rmse_list.append(np.sqrt(sk_mse))


# Υπολογισμός μέσης τιμής και τυπικής απόκλισης
mean_rmse = np.mean(rmse_list)
std_rmse = np.std(rmse_list)

print(f"Μέση τιμή RMSE 20 φορές: {mean_rmse:.4f}")
print(f"Τυπική απόκλιση RMSE 20 φορές: {std_rmse:.4f}")

sk_mean_rmse = np.mean(sk_rmse_list)
sk_std_rmse = np.std(sk_rmse_list)

print(f"Μέση τιμή RMSE 20 φορές: {sk_mean_rmse:.4f}")
print(f"Τυπική απόκλιση RMSE 20 φορές: {sk_std_rmse:.4f}")
