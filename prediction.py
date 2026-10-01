from pathlib import Path

import joblib
import pandas as pd
import numpy as np
import shap

ARTIFACT_PATH = Path(
    "artifacts/churn_model.joblib"
)

model_artifact = joblib.load(
    ARTIFACT_PATH
)

pipeline = model_artifact["pipeline"]
threshold = model_artifact["threshold"]

preprocessor = (
    pipeline.named_steps["preprocessor"]
)

model = (
    pipeline.named_steps["model"]
)

explainer = shap.TreeExplainer(
    model
)

def get_original_feature_name(
    feature_name: str
) -> str:
    transformed_name = feature_name.split(
        "__",
        maxsplit=1
    )[1]

    if feature_name.startswith("num__"):
        return transformed_name

    return transformed_name.split(
        "_",
        maxsplit=1
    )[0]

def predict_customer(customer_data: dict) -> dict:
    customer_frame = pd.DataFrame(
        [customer_data]
    )

    processed_customer = (
        preprocessor.transform(
            customer_frame
        )
    )

    explanation = explainer(
        processed_customer
    )

    feature_names = (
        preprocessor.get_feature_names_out()
    )

    shap_values = explanation.values[0]

    shap_table = pd.DataFrame(
        {
            "encoded_feature": feature_names,
            "shap_value": shap_values
        }
    )

    shap_table["original_feature"] = (
        shap_table["encoded_feature"].apply(
            get_original_feature_name
        )
    )

    grouped_shap = (
        shap_table
        .groupby("original_feature")["shap_value"]
        .sum()
        .sort_values(
            key=lambda values: values.abs(),
            ascending=False
        )
        .head(5)
    )
    probability = float(
        pipeline.predict_proba(
            customer_frame
        )[0, 1]
    )

    predicted_class = int(
        probability >= threshold
    )

    prediction_label = (
        "Churn"
        if predicted_class == 1
        else "No churn"
    )

    return {
    "churn_probability": round(
        probability,
        4
    ),
    "predicted_class": predicted_class,
    "prediction_label": prediction_label,
    "threshold": threshold,
    "top_factors": [
    {
        "feature": feature_name,
        "value": customer_data[feature_name],
        "impact": (
            "increases churn risk"
            if shap_value > 0
            else "decreases churn risk"
        ),
        "shap_value": round(
            float(shap_value),
            4
        )
    }
    for feature_name, shap_value
    in grouped_shap.items()
]
}

if __name__ == "__main__":
    example_customer = {
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

    result = predict_customer(
        example_customer
    )

    print(result)