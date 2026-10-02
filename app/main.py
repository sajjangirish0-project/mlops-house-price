from fastapi import FastAPI
import joblib

app = FastAPI()

model = joblib.load("model/model.pkl")


@app.get("/")
def home():
    return {
        "message": "House Price Prediction API"
    }


@app.post("/predict")
def predict(
    area: float,
    bedrooms: int,
    bathrooms: int,
    age: int
):

    prediction = model.predict([
        [area, bedrooms, bathrooms, age]
    ])

    return {
        "predicted_price": prediction[0]
    }