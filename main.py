from pydantic import BaseModel
from fastapi import FastAPI
import pandas as pd
import joblib

# Expected input
class Home_Credit(BaseModel):
    EXT_SOURCE_1: float
    NAME_EDUCATION_TYPE: int
    AMT_CREDIT: float
    Max_DURATION_DUE_VERSION: float
    YEARS_EMPLOYED: float
    EXT_SOURCE_3: float
    YEARS_BIRTH: float
    Max_DURATION_DECISION_DRAWING: float
    EXT_SOURCE_2: float
    Min_RATIO_GOODS_PRICE_CREDIT: float
    AVG_Risk_Score: float
    Min_DURATION_DECISION_DRAWING: float
    CODE_GENDER_F: float
    YEARS_LAST_PHONE_CHANGE: float


# Expected output
class PredictionOut(BaseModel):
    default_proba: float


#model = GaussianNB(var_smoothing=1e-09)
model = joblib.load("model.pkl")

app = FastAPI()


# Home page
@app.get("/")
def home():
    return {"message": "HomeCredit Default App", "model_version": 0.1}



# Inference endpoint
@app.post("/predict", response_model=PredictionOut)
def predict(payload: Home_Credit):
    cust_df = pd.DataFrame([payload.model_dump()])
    predictions = model.predict_proba(cust_df)[0, 1]

    adjusted_predictions = (predictions > 0.12).astype(int)

    result = {"default_proba": adjusted_predictions}
    return result