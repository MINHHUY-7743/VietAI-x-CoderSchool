import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# from odf.svg import Lineargradient
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV

file_path = 'web-traffic.csv'
data = pd.read_csv(file_path)

data["date"] = pd.to_datetime(data["date"], format='%d/%m/%y', errors='coerce')
# print(data.info())

data["date"] = data["date"].interpolate()


def create_ts_data(data, window_size):
    i = 1
    while i < window_size:
      data["user_{}".format(i)] = data["users"].shift(-i)
      i += 1
    data["target"] = data["users"].shift(-i)
    data = data.dropna(axis=0)
    return data
window_size = 5
train_ratio = 0.8
data = create_ts_data(data, window_size)


x = data.drop(columns=["date", "target"])
y = data["target"]
num_samples = len(data)
x_train = x[:int(num_samples*train_ratio)]
y_train = y[:int(num_samples*train_ratio)]
x_test = x[int(num_samples*train_ratio):]
y_test = y[int(num_samples*train_ratio):]

# model = LinearRegression()
model = Pipeline(steps=[
    ("scaler", StandardScaler()),# du thua
    #("reg", LinearRegression())
    ("reg", RandomForestRegressor(n_estimators=100, random_state=42))
])

param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [10, 20, 30, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
}
rf = RandomForestRegressor(random_state=42)
grid_search = GridSearchCV(
    estimator=rf,
    param_grid=param_grid,
    scoring='r2',
    cv=3,
    verbose=2,
    n_jobs=-1
)
# model.fit(x_train, y_train)
grid_search.fit(x_train, y_train)

print("Best Parameters:", grid_search.best_params_)
print("Best R2 Score:", grid_search.best_score_)

# y_predict = model.predict(x_test)
# for i, j in zip(y_predict, y_test):
#   print("predict value: {}. Actual value: {}".format(i,j))
best_rf = grid_search.best_estimator_
y_predict = best_rf.predict(x_test)

print("MAE: {}".format(mean_absolute_error(y_test, y_predict)))
print("MSE: {}".format(mean_squared_error(y_test, y_predict)))
print("R2: {}".format(r2_score(y_test, y_predict)))


fig, ax = plt.subplots()
ax.plot(data["date"][:int(num_samples*train_ratio)], y_train, label = "train")
ax.plot(data["date"][int(num_samples*train_ratio):], y_test, label = "test")
ax.plot(data["date"][int(num_samples*train_ratio):], y_predict, label = "prediction")
ax.set_xlabel("date")
ax.set_ylabel("users")
ax.legend()
ax.grid()
plt.show()
