import pandas as pd
import joblib
import mlflow
import mlflow.sklearn

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

df = pd.read_csv("data/house_prices.csv")

X = df[["area", "bedrooms", "bathrooms", "age"]]
y = df["price"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LinearRegression()

mlflow.set_experiment("house-price-prediction")

with mlflow.start_run():

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    mse = mean_squared_error(y_test, predictions)

    mlflow.log_metric("mse", mse)

    mlflow.sklearn.log_model(
        model,
        "model"
    )

    joblib.dump(model, "model/model.pkl")

    print("Model trained successfully")
    print("MSE:", mse)