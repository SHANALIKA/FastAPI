# main.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import numpy as np

app = FastAPI()

# Load trained model
model = joblib.load("model.pkl")
target_names = ["setosa", "versicolor", "virginica"]

# ------------------
# Existing endpoints
# ------------------

@app.get("/")
def read_root():
    return {"message": "Hello, FastAPI!"}

@app.get("/items/{item_id}")
def read_item(item_id: int, q: str = None):
    return {"item_id": item_id, "q": q}

# ------------------
# New: Prediction endpoint
# ------------------

# Pydantic model for input validation
class IrisFeatures(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float

@app.post("/predict")
def predict(features: IrisFeatures):
    try:
        data = np.array([[features.sepal_length,
                          features.sepal_width,
                          features.petal_length,
                          features.petal_width]])
        pred = model.predict(data)[0]
        return {"prediction": target_names[pred]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
