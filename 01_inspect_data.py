from pathlib import Path
import pandas as pd
import shap
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix
)
from xgboost import XGBClassifier
from sklearn.model_selection import (
    StratifiedKFold,
    cross_validate
)

from sklearn.pipeline import Pipeline
import numpy as np
import joblib

data_path = Path("data/raw/Telco-Customer-Churn.csv")

df = pd.read_csv(data_path)

print(df.head())
print("shape", df.shape)
print("\nColumns:")
print(df.columns.to_list())
print("info")
print(df.info())
print("\nblank string count by column:")
print(df.apply(lambda column: (column == " ").sum()))
blank_total_charges = df["TotalCharges"] == " "

print("\nRows with blank TotalCharges:")
print(
    df.loc[
        blank_total_charges,
        ["customerID", "tenure", "TotalCharges", "Churn"]
    ]
)

df["TotalCharges"] = pd.to_numeric(
    df["TotalCharges"],
    errors="coerce"
)

print("\nMissing TotalCharges after numeric conversion:")
print(df["TotalCharges"].isna().sum())

print("\nTotalCharges data type:")
print(df["TotalCharges"].dtype)
df["TotalCharges"] = df["TotalCharges"].fillna(0)

print("\nMissing TotalCharges after filling:")
print(df["TotalCharges"].isna().sum())

print("\nMissing values by column:")

missing_values = df.isna().sum()

print(missing_values[missing_values > 0])

print("\nduplicate rows:")
print(df.duplicated().sum())

print("\nUnique customer IDs:")
print(df["customerID"].nunique())

print("\nTotal rows:")
print(len(df))

print("\nChurn counts:")
print(df["Churn"].value_counts())

print("\nChurn percentages:")
print(df["Churn"].value_counts(normalize=True).mul(100).round(2))

print("\nNumerical column summary:")
print(df.describe())

print("\nNumerical features grouped by churn:")
print(
    df.groupby("Churn")[
        ["SeniorCitizen", "tenure", "MonthlyCharges", "TotalCharges"]
    ].mean().round(2)
)

print("\nChurn percentage by contract type:")
print(
    pd.crosstab(
        df["Contract"],
        df["Churn"],
        normalize="index"
    ).mul(100).round(2)
)

print("\nChurn percentage by payment method:")
print(
    pd.crosstab(
        df["PaymentMethod"],
        df["Churn"],
        normalize="index"
    ).mul(100).round(2)
)

print("\nChurn percentage by internet service:")
print(
    pd.crosstab(
        df["InternetService"],
        df["Churn"],
        normalize="index"
    ).mul(100).round(2)
)

df["Churn"] = df["Churn"].map({
    "No": 0,
    "Yes": 1
})

print("\nEncoded target values:")
print(df["Churn"].value_counts())

X = df.drop(columns=["Churn", "customerID"])
y = df["Churn"]

print("\nFeature shape:")
print(X.shape)

print("\nTarget shape:")
print(y.shape)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\nTraining feature shape:")
print(X_train.shape)

print("\nTesting feature shape:")
print(X_test.shape)

print("\nTraining target distribution:")
print(y_train.value_counts(normalize=True).round(3))

print("\nTesting target distribution:")
print(y_test.value_counts(normalize=True).round(3))

numeric_features = X.select_dtypes(include="number").columns.tolist()

categorical_features = X.select_dtypes(
    exclude="number"
).columns.tolist()

print("\nNumerical features:")
print(numeric_features)

print("\nCategorical features:")
print(categorical_features)

preprocessor = ColumnTransformer(
    transformers=[
        (
            "num",
            StandardScaler(),
            numeric_features
        ),
        (
            "cat",
            OneHotEncoder(handle_unknown="ignore"),
            categorical_features
        )
    ]
)

X_train_processed = preprocessor.fit_transform(X_train)

X_test_processed = preprocessor.transform(X_test)

print("\nProcessed training shape:")
print(X_train_processed.shape)

print("\nProcessed testing shape:")
print(X_test_processed.shape)

feature_names = preprocessor.get_feature_names_out()

print("\nNumber of processed features:")
print(len(feature_names))

print("\nProcessed feature names:")
print(feature_names)

model = LogisticRegression(max_iter=1000)

model.fit(X_train_processed, y_train)

print("\nModel training completed.")

y_pred = model.predict(X_test_processed)

y_probability = model.predict_proba(X_test_processed)[:, 1]


print("\nFirst 10 predicted classes:")
print(y_pred[:10])

print("\nFirst 10 churn probabilities:")
print(y_probability[:10].round(3))

accuracy = accuracy_score(y_test, y_pred)

precision = precision_score(y_test, y_pred)

recall = recall_score(y_test, y_pred)

f1 = f1_score(y_test, y_pred)

roc_auc = roc_auc_score(y_test, y_probability)

matrix = confusion_matrix(y_test, y_pred)

print("\nBaseline model metrics:")
print(f"Accuracy:  {accuracy:.3f}")
print(f"Precision: {precision:.3f}")
print(f"Recall:    {recall:.3f}")
print(f"F1-score:  {f1:.3f}")
print(f"ROC-AUC:   {roc_auc:.3f}")

print("\nConfusion matrix:")
print(matrix)

balanced_model = LogisticRegression(
    max_iter=1000,
    class_weight="balanced"
)

balanced_model.fit(
    X_train_processed,
    y_train
)

balanced_pred = balanced_model.predict(X_test_processed)

balanced_probability = balanced_model.predict_proba(
    X_test_processed
)[:, 1]

balanced_accuracy = accuracy_score(
    y_test,
    balanced_pred
)

balanced_precision = precision_score(
    y_test,
    balanced_pred
)

balanced_recall = recall_score(
    y_test,
    balanced_pred
)

balanced_f1 = f1_score(
    y_test,
    balanced_pred
)

balanced_roc_auc = roc_auc_score(
    y_test,
    balanced_probability
)

balanced_matrix = confusion_matrix(
    y_test,
    balanced_pred
)

print("\nBalanced model metrics:")
print(f"Accuracy:  {balanced_accuracy:.3f}")
print(f"Precision: {balanced_precision:.3f}")
print(f"Recall:    {balanced_recall:.3f}")
print(f"F1-score:  {balanced_f1:.3f}")
print(f"ROC-AUC:   {balanced_roc_auc:.3f}")

print("\nBalanced model confusion matrix:")
print(balanced_matrix)

print("\nThreshold comparison:")

for threshold in [0.30, 0.40, 0.50, 0.60]:
    threshold_pred = (
        balanced_probability >= threshold
    ).astype(int)

    threshold_precision = precision_score(
        y_test,
        threshold_pred,
        zero_division=0
    )

    threshold_recall = recall_score(
        y_test,
        threshold_pred,
        zero_division=0
    )

    threshold_f1 = f1_score(
        y_test,
        threshold_pred,
        zero_division=0
    )

    print(
        f"Threshold: {threshold:.2f} | "
        f"Precision: {threshold_precision:.3f} | "
        f"Recall: {threshold_recall:.3f} | "
        f"F1: {threshold_f1:.3f}"
    )

xgb_model = XGBClassifier(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=4,
    subsample=0.8,
    colsample_bytree=0.8,
    eval_metric="logloss",
    random_state=42
)

xgb_model.fit(
    X_train_processed,
    y_train
)

print("\nXGBoost training completed.")

xgb_pred = xgb_model.predict(X_test_processed)

xgb_probability = xgb_model.predict_proba(
    X_test_processed
)[:, 1]

xgb_accuracy = accuracy_score(
    y_test,
    xgb_pred
)

xgb_precision = precision_score(
    y_test,
    xgb_pred
)

xgb_recall = recall_score(
    y_test,
    xgb_pred
)

xgb_f1 = f1_score(
    y_test,
    xgb_pred
)

xgb_roc_auc = roc_auc_score(
    y_test,
    xgb_probability
)

xgb_matrix = confusion_matrix(
    y_test,
    xgb_pred
)

print("\nXGBoost metrics:")
print(f"Accuracy:  {xgb_accuracy:.3f}")
print(f"Precision: {xgb_precision:.3f}")
print(f"Recall:    {xgb_recall:.3f}")
print(f"F1-score:  {xgb_f1:.3f}")
print(f"ROC-AUC:   {xgb_roc_auc:.3f}")

print("\nXGBoost confusion matrix:")
print(xgb_matrix)

negative_count = (y_train == 0).sum()
positive_count = (y_train == 1).sum()

scale_pos_weight = negative_count / positive_count

print("\nXGBoost scale_pos_weight:")
print(scale_pos_weight)

xgb_balanced_model = XGBClassifier(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=4,
    subsample=0.8,
    colsample_bytree=0.8,
    scale_pos_weight=scale_pos_weight,
    eval_metric="logloss",
    random_state=42
)

xgb_balanced_model.fit(
    X_train_processed,
    y_train
)

print("\nBalanced XGBoost training completed.")

xgb_balanced_pred = xgb_balanced_model.predict(
    X_test_processed
)

xgb_balanced_probability = xgb_balanced_model.predict_proba(
    X_test_processed
)[:, 1]

xgb_balanced_accuracy = accuracy_score(
    y_test,
    xgb_balanced_pred
)

xgb_balanced_precision = precision_score(
    y_test,
    xgb_balanced_pred
)

xgb_balanced_recall = recall_score(
    y_test,
    xgb_balanced_pred
)

xgb_balanced_f1 = f1_score(
    y_test,
    xgb_balanced_pred
)

xgb_balanced_roc_auc = roc_auc_score(
    y_test,
    xgb_balanced_probability
)

xgb_balanced_matrix = confusion_matrix(
    y_test,
    xgb_balanced_pred
)

print("\nBalanced XGBoost metrics:")
print(f"Accuracy:  {xgb_balanced_accuracy:.3f}")
print(f"Precision: {xgb_balanced_precision:.3f}")
print(f"Recall:    {xgb_balanced_recall:.3f}")
print(f"F1-score:  {xgb_balanced_f1:.3f}")
print(f"ROC-AUC:   {xgb_balanced_roc_auc:.3f}")

print("\nBalanced XGBoost confusion matrix:")
print(xgb_balanced_matrix)

print("\nBalanced XGBoost threshold comparison:")

for threshold in [0.30, 0.40, 0.50, 0.60]:
    xgb_threshold_pred = (
        xgb_balanced_probability >= threshold
    ).astype(int)

    threshold_precision = precision_score(
        y_test,
        xgb_threshold_pred,
        zero_division=0
    )

    threshold_recall = recall_score(
        y_test,
        xgb_threshold_pred,
        zero_division=0
    )

    threshold_f1 = f1_score(
        y_test,
        xgb_threshold_pred,
        zero_division=0
    )

    print(
        f"Threshold: {threshold:.2f} | "
        f"Precision: {threshold_precision:.3f} | "
        f"Recall: {threshold_recall:.3f} | "
        f"F1: {threshold_f1:.3f}"
    )

balanced_xgb_pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        (
            "model",
            XGBClassifier(
                n_estimators=300,
                learning_rate=0.05,
                max_depth=4,
                subsample=0.8,
                colsample_bytree=0.8,
                scale_pos_weight=scale_pos_weight,
                eval_metric="logloss",
                random_state=42
            )
        )
    ]
)

cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

cv_results = cross_validate(
    balanced_xgb_pipeline,
    X_train,
    y_train,
    cv=cv,
    scoring={
        "precision": "precision",
        "recall": "recall",
        "f1": "f1",
        "roc_auc": "roc_auc"
    },
    n_jobs=-1
)

print("\nCross-validation results:")

for metric in ["precision", "recall", "f1", "roc_auc"]:
    scores = cv_results[f"test_{metric}"]

    print(
        f"{metric}: "
        f"mean={scores.mean():.3f}, "
        f"std={scores.std():.3f}"
    )

explainer = shap.TreeExplainer(
    xgb_balanced_model
)

sample_index = 0

sample = X_test_processed[
    sample_index:sample_index + 1
]

sample_explanation = explainer(
    sample
)

print("\nSHAP values shape:")
print(sample_explanation.values.shape)

print("\nBase value:")
print(sample_explanation.base_values)

print("\nModel churn probability:")
print(xgb_balanced_probability[sample_index])

sample_shap_values = sample_explanation.values[0]

feature_contributions = pd.DataFrame(
    {
        "feature": feature_names,
        "shap_value": sample_shap_values
    }
)

feature_contributions["absolute_shap"] = (
    feature_contributions["shap_value"].abs()
)

top_contributions = feature_contributions.sort_values(
    "absolute_shap",
    ascending=False
).head(10)

print("\nTop feature contributions:")
print(
    top_contributions[
        ["feature", "shap_value"]
    ].to_string(index=False)
)

print("\nOriginal customer features:")
print(X_test.iloc[sample_index].to_string())

print("\nFeatures pushing toward churn:")

print(
    feature_contributions[
        feature_contributions["shap_value"] > 0
    ]
    .sort_values("shap_value", ascending=False)
    .head(5)
    [["feature", "shap_value"]]
    .to_string(index=False)
)

print("\nFeatures pushing away from churn:")

print(
    feature_contributions[
        feature_contributions["shap_value"] < 0
    ]
    .sort_values("shap_value")
    .head(5)
    [["feature", "shap_value"]]
    .to_string(index=False)
)

all_explanations = explainer(
    X_test_processed
)

mean_absolute_shap = np.abs(
    all_explanations.values
).mean(axis=0)

global_importance = pd.DataFrame(
    {
        "feature": feature_names,
        "mean_absolute_shap": mean_absolute_shap
    }
).sort_values(
    "mean_absolute_shap",
    ascending=False
)

print("\nGlobal feature importance:")
print(
    global_importance.head(15).to_string(
        index=False
    )
)

def get_original_feature_name(feature_name):
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

global_importance["original_feature"] = (
    global_importance["feature"].apply(
        get_original_feature_name
    )
)

grouped_importance = (
    global_importance
    .groupby("original_feature")[
        "mean_absolute_shap"
    ]
    .sum()
    .sort_values(
        ascending=False
    )
)

print("\nBusiness-level global feature importance:")
print(grouped_importance.head(10))

logistic_pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        (
            "model",
            LogisticRegression(
                max_iter=1000
            )
        )
    ]
)

balanced_logistic_pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        (
            "model",
            LogisticRegression(
                max_iter=1000,
                class_weight="balanced"
            )
        )
    ]
)

xgb_pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        (
            "model",
            XGBClassifier(
                n_estimators=300,
                learning_rate=0.05,
                max_depth=4,
                subsample=0.8,
                colsample_bytree=0.8,
                eval_metric="logloss",
                random_state=42
            )
        )
    ]
)

balanced_xgb_pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        (
            "model",
            XGBClassifier(
                n_estimators=300,
                learning_rate=0.05,
                max_depth=4,
                subsample=0.8,
                colsample_bytree=0.8,
                scale_pos_weight=scale_pos_weight,
                eval_metric="logloss",
                random_state=42
            )
        )
    ]
)

candidate_models = {
    "Logistic Regression": logistic_pipeline,
    "Balanced Logistic Regression": balanced_logistic_pipeline,
    "XGBoost": xgb_pipeline,
    "Balanced XGBoost": balanced_xgb_pipeline
}

print("\nCross-validation model comparison:")

for model_name, candidate_model in candidate_models.items():
    results = cross_validate(
        candidate_model,
        X_train,
        y_train,
        cv=cv,
        scoring={
            "precision": "precision",
            "recall": "recall",
            "f1": "f1",
            "roc_auc": "roc_auc"
        },
        n_jobs=-1
    )

    print(f"\n{model_name}")
    print(
        f"Precision: "
        f"{results['test_precision'].mean():.3f} "
        f"+/- {results['test_precision'].std():.3f}"
    )
    print(
        f"Recall: "
        f"{results['test_recall'].mean():.3f} "
        f"+/- {results['test_recall'].std():.3f}"
    )
    print(
        f"F1-score: "
        f"{results['test_f1'].mean():.3f} "
        f"+/- {results['test_f1'].std():.3f}"
    )
    print(
        f"ROC-AUC: "
        f"{results['test_roc_auc'].mean():.3f} "
        f"+/- {results['test_roc_auc'].std():.3f}"
    )

    balanced_xgb_pipeline.fit(
    X_train,
    y_train
)

final_test_pred = balanced_xgb_pipeline.predict(
    X_test
)

final_test_probability = (
    balanced_xgb_pipeline.predict_proba(
        X_test
    )[:, 1]
)

print("\nFinal pipeline training completed.")

final_accuracy = accuracy_score(
    y_test,
    final_test_pred
)

final_precision = precision_score(
    y_test,
    final_test_pred
)

final_recall = recall_score(
    y_test,
    final_test_pred
)

final_f1 = f1_score(
    y_test,
    final_test_pred
)

final_roc_auc = roc_auc_score(
    y_test,
    final_test_probability
)

final_matrix = confusion_matrix(
    y_test,
    final_test_pred
)

print("\nFinal pipeline metrics:")
print(f"Accuracy:  {final_accuracy:.3f}")
print(f"Precision: {final_precision:.3f}")
print(f"Recall:    {final_recall:.3f}")
print(f"F1-score:  {final_f1:.3f}")
print(f"ROC-AUC:   {final_roc_auc:.3f}")

print("\nFinal pipeline confusion matrix:")
print(final_matrix)

artifacts_path = Path("artifacts")
artifacts_path.mkdir(exist_ok=True)

model_artifact = {
    "pipeline": balanced_xgb_pipeline,
    "threshold": 0.50,
    "feature_columns": X.columns.tolist()
}

joblib.dump(
    model_artifact,
    artifacts_path / "churn_model.joblib"
)

print("\nModel artifact saved successfully.")

loaded_artifact = joblib.load(
    "artifacts/churn_model.joblib"
)

loaded_pipeline = loaded_artifact["pipeline"]
loaded_threshold = loaded_artifact["threshold"]

one_customer = X_test.iloc[[0]]

loaded_probability = loaded_pipeline.predict_proba(
    one_customer
)[0, 1]

loaded_prediction = int(
    loaded_probability >= loaded_threshold
)

print("\nLoaded model prediction:")
print(f"Churn probability: {loaded_probability:.3f}")
print(f"Predicted class: {loaded_prediction}")
print(f"Threshold used: {loaded_threshold}")

new_customer = pd.DataFrame(
    [
        {
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
    ]
)

new_probability = loaded_pipeline.predict_proba(
    new_customer
)[0, 1]

new_prediction = int(
    new_probability >= loaded_threshold
)

risk_level = (
    "High"
    if new_probability >= 0.50
    else "Low"
)

print("\nNew customer prediction:")
print(f"Churn probability: {new_probability:.3f}")
print(f"Predicted class: {new_prediction}")
print(f"Risk level: {risk_level}")

loaded_preprocessor = (
    loaded_pipeline.named_steps["preprocessor"]
)

loaded_model = (
    loaded_pipeline.named_steps["model"]
)

new_customer_processed = (
    loaded_preprocessor.transform(new_customer)
)

new_explainer = shap.TreeExplainer(
    loaded_model
)

new_explanation = new_explainer(
    new_customer_processed
)

new_feature_names = (
    loaded_preprocessor.get_feature_names_out()
)

new_shap_values = new_explanation.values[0]

new_contributions = pd.DataFrame(
    {
        "feature": new_feature_names,
        "shap_value": new_shap_values
    }
)

new_contributions["absolute_shap"] = (
    new_contributions["shap_value"].abs()
)

print("\nTop factors for the new customer:")
print(
    new_contributions
    .sort_values(
        "absolute_shap",
        ascending=False
    )
    .head(10)
    [["feature", "shap_value"]]
    .to_string(index=False)
)