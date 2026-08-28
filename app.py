import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel

# 1. Load the pre-trained pipeline model
# This contains both the preprocessor transformers and the tuned XGBoost model
model = joblib.load('churn_pipeline_model.pkl')

# 2. Initialize FastAPI app
app = FastAPI(
    title="Customer Churn Prediction API",
    description="API to predict customer churn probability using an optimized XGBoost pipeline.",
    version="1.0"
)

# 3. Define the expected incoming data structure using Pydantic
# The field names must exactly match the columns your original raw dataframe had
class CustomerData(BaseModel):
    gender: str
    SeniorCitizen: int
    Partner: str
    Dependents: str
    tenure: int
    PhoneService: str
    MultipleLines: str
    InternetService: str
    OnlineSecurity: str
    OnlineBackup: str
    DeviceProtection: str
    TechSupport: str
    StreamingTV: str
    StreamingMovies: str
    Contract: str
    PaperlessBilling: str
    PaymentMethod: str
    MonthlyCharges: float
    TotalCharges: float


@app.get("/")
def home():
    return {"message": "Churn Prediction API is online. Go to /docs for interactive testing."}


@app.post("/predict")
def predict_churn(customer: CustomerData):
    # Convert incoming Pydantic payload into a standard dictionary
    data_dict = customer.model_dump()
    
    # Convert the dictionary into a 1-row Pandas DataFrame for the pipeline
    input_df = pd.DataFrame([data_dict])
    
    # Get raw prediction probabilities
    # index 1 gives the probability of class 1 (Churn)
    churn_probability = float(model.predict_proba(input_df)[0][1])
    
    # Apply our custom tuned decision threshold (from main.py optimization)
    custom_threshold = 0.4
    prediction = 1 if churn_probability >= custom_threshold else 0
    
    return {
        "churn_probability": round(churn_probability, 4),
        "prediction_label": prediction,
        "status": "High Churn Risk - Trigger Retention Campaign" if prediction == 1 else "Low Risk"
    }