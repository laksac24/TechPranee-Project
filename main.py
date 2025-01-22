from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
import joblib
import os

app = FastAPI()

model = None
data = None

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

class PredictInput(BaseModel):
    Temperature: float
    Run_Time: float

@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    """Endpoint to upload a CSV file containing the dataset."""
    global data

    if not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only CSV files are allowed.")

    file_path = os.path.join(UPLOAD_DIR, file.filename)
    with open(file_path, "wb") as f:
        f.write(await file.read())

    data = pd.read_csv(file_path)

    required_columns = {"Temperature", "Run_Time", "Downtime_Flag"}
    if not required_columns.issubset(data.columns):
        raise HTTPException(
            status_code=400,
            detail=f"Dataset must contain columns: {', '.join(required_columns)}",
        )

    return {"message": "File uploaded and dataset loaded successfully."}

@app.post("/train")
def train_model():
    """Endpoint to train the ML model on the uploaded dataset."""
    global model, data

    if data is None:
        raise HTTPException(status_code=400, detail="No dataset uploaded.")

    X = data[["Temperature", "Run_Time"]]
    y = data["Downtime_Flag"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = LogisticRegression()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    model_path = os.path.join(UPLOAD_DIR, "model.joblib")
    joblib.dump(model, model_path)

    return {"message": "Model trained successfully.", "accuracy": accuracy, "f1_score": f1}

@app.post("/predict")
def predict(input_data: PredictInput):
    """Endpoint to make a prediction using the trained model."""
    global model

    if model is None:
        raise HTTPException(status_code=400, detail="Model not trained.")

    features = [[input_data.Temperature, input_data.Run_Time]]
    prediction = model.predict(features)[0]
    confidence = model.predict_proba(features)[0].max()

    return {"Downtime": "Yes" if prediction == 1 else "No", "Confidence": round(confidence, 2)}


