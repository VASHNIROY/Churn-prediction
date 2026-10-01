from fastapi import FastAPI
from pydantic import BaseModel, Field
from prediction import predict_customer
from fastapi.middleware.cors import CORSMiddleware

class CustomerRequest(BaseModel):
    gender: str
    SeniorCitizen: int = Field(
        ge=0,
        le=1
    )
    Partner: str
    Dependents: str
    tenure: int = Field(
    ge=0,
    le=72
    )
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
    MonthlyCharges: float = Field(
    gt=0
    )
    TotalCharges: float = Field(
    ge=0
    )

app = FastAPI(
    title="Customer Churn Prediction API",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

@app.get("/health")
def health_check():
    return {
        "status": "healthy"
    }

@app.post("/predict")
def predict_churn(
    customer: CustomerRequest
):
    customer_data = customer.model_dump()

    return predict_customer(
        customer_data
    )