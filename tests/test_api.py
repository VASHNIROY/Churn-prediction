from fastapi.testclient import TestClient

from main import app


client = TestClient(app)


valid_customer = {
    "gender": "Female",
    "SeniorCitizen": 0,
    "Partner": "No",
    "Dependents": "No",
    "tenure": 3,
    "PhoneService": "Yes",
    "MultipleLines": "No",
    "InternetService": "Fiber optic",
    "OnlineSecurity": "No",
    "OnlineBackup": "No",
    "DeviceProtection": "No",
    "TechSupport": "No",
    "StreamingTV": "Yes",
    "StreamingMovies": "Yes",
    "Contract": "Month-to-month",
    "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check",
    "MonthlyCharges": 95.0,
    "TotalCharges": 285.0
}


def test_health_endpoint():
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {
        "status": "healthy"
    }


def test_valid_prediction():
    response = client.post(
        "/predict",
        json=valid_customer
    )

    assert response.status_code == 200

    body = response.json()

    assert "churn_probability" in body
    assert "prediction_label" in body
    assert "top_factors" in body


def test_invalid_tenure():
    invalid_customer = {
        **valid_customer,
        "tenure": -5
    }

    response = client.post(
        "/predict",
        json=invalid_customer
    )

    assert response.status_code == 422


def test_missing_required_field():
    incomplete_customer = {
        key: value
        for key, value in valid_customer.items()
        if key != "tenure"
    }

    response = client.post(
        "/predict",
        json=incomplete_customer
    )

    assert response.status_code == 422