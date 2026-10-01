import { useState, type SubmitEvent as ReactSubmitEvent } from "react";
import "./App.css";

const API_URL = import.meta.env.VITE_API_BASE_URL ?? "http://127.0.0.1:8000";

type CustomerForm = {
  gender: string;
  SeniorCitizen: number | "";
  Partner: string;
  Dependents: string;
  tenure: number | "";
  PhoneService: string;
  MultipleLines: string;
  InternetService: string;
  OnlineSecurity: string;
  OnlineBackup: string;
  DeviceProtection: string;
  TechSupport: string;
  StreamingTV: string;
  StreamingMovies: string;
  Contract: string;
  PaperlessBilling: string;
  PaymentMethod: string;
  MonthlyCharges: number | "";
  TotalCharges: number | "";
};

type PredictionFactor = {
  feature: string;
  value: string | number;
  impact: string;
  shap_value: number;
};

type PredictionResult = {
  churn_probability: number;
  predicted_class: number;
  prediction_label: string;
  threshold: number;
  top_factors: PredictionFactor[];
};

const initialCustomer: CustomerForm = {
  gender: "Female",
  SeniorCitizen: 0,
  Partner: "No",
  Dependents: "No",
  tenure: 3,
  PhoneService: "Yes",
  MultipleLines: "No",
  InternetService: "Fiber optic",
  OnlineSecurity: "No",
  OnlineBackup: "No",
  DeviceProtection: "No",
  TechSupport: "No",
  StreamingTV: "Yes",
  StreamingMovies: "Yes",
  Contract: "Month-to-month",
  PaperlessBilling: "Yes",
  PaymentMethod: "Electronic check",
  MonthlyCharges: 95,
  TotalCharges: 285,
};

const selectOptions: Record<string, string[]> = {
  gender: ["Female", "Male"],
  Partner: ["Yes", "No"],
  Dependents: ["Yes", "No"],
  PhoneService: ["Yes", "No"],
  MultipleLines: ["Yes", "No", "No phone service"],
  InternetService: ["DSL", "Fiber optic", "No"],
  OnlineSecurity: ["Yes", "No", "No internet service"],
  OnlineBackup: ["Yes", "No", "No internet service"],
  DeviceProtection: ["Yes", "No", "No internet service"],
  TechSupport: ["Yes", "No", "No internet service"],
  StreamingTV: ["Yes", "No", "No internet service"],
  StreamingMovies: ["Yes", "No", "No internet service"],
  Contract: ["Month-to-month", "One year", "Two year"],
  PaperlessBilling: ["Yes", "No"],
  PaymentMethod: [
    "Electronic check",
    "Mailed check",
    "Bank transfer (automatic)",
    "Credit card (automatic)",
  ],
};

const fieldLabels: Record<keyof CustomerForm, string> = {
  gender: "Gender",
  SeniorCitizen: "Senior citizen",
  Partner: "Partner",
  Dependents: "Dependents",
  tenure: "Tenure in months",
  PhoneService: "Phone service",
  MultipleLines: "Multiple lines",
  InternetService: "Internet service",
  OnlineSecurity: "Online security",
  OnlineBackup: "Online backup",
  DeviceProtection: "Device protection",
  TechSupport: "Technical support",
  StreamingTV: "Streaming TV",
  StreamingMovies: "Streaming movies",
  Contract: "Contract",
  PaperlessBilling: "Paperless billing",
  PaymentMethod: "Payment method",
  MonthlyCharges: "Monthly charges",
  TotalCharges: "Total charges",
};

const numericFields: Array<keyof CustomerForm> = [
  "SeniorCitizen",
  "tenure",
  "MonthlyCharges",
  "TotalCharges",
];

function App() {
  const [customer, setCustomer] = useState<CustomerForm>(initialCustomer);

  const [result, setResult] = useState<PredictionResult | null>(null);

  const [loading, setLoading] = useState(false);

  const [error, setError] = useState<string | null>(null);
  function updateField(field: keyof CustomerForm, value: string) {
    const finalValue = numericFields.includes(field)
      ? value === ""
        ? ""
        : Number(value)
      : value;

    setCustomer(
      (current) =>
        ({
          ...current,
          [field]: finalValue,
        }) as CustomerForm,
    );
  }

  function resetForm() {
    setCustomer(initialCustomer);
    setResult(null);
    setError(null);
  }

  async function handleSubmit(event: ReactSubmitEvent<HTMLFormElement>) {
    event.preventDefault();

    setLoading(true);
    setError(null);
    setResult(null);

    const controller = new AbortController();

    const timeout = window.setTimeout(() => controller.abort(), 15_000);

    try {
      const numericValues = [
        customer.SeniorCitizen,
        customer.tenure,
        customer.MonthlyCharges,
        customer.TotalCharges,
      ];

      if (numericValues.some((value) => value === "")) {
        setError("Please enter all numerical customer values.");
        setLoading(false);
        return;
      }
      const payload = {
        ...customer,
        SeniorCitizen: Number(customer.SeniorCitizen),
        tenure: Number(customer.tenure),
        MonthlyCharges: Number(customer.MonthlyCharges),
        TotalCharges: Number(customer.TotalCharges),
      };
      const response = await fetch(`${API_URL}/predict`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify(payload),
        signal: controller.signal,
      });

      const responseBody = await response.json();

      if (!response.ok) {
        throw new Error(responseBody.detail ?? "Prediction request failed.");
      }

      setResult(responseBody);
    } catch (requestError) {
      if (
        requestError instanceof DOMException &&
        requestError.name === "AbortError"
      ) {
        setError("The prediction request timed out.");
      } else {
        setError(
          requestError instanceof Error
            ? requestError.message
            : "Unable to generate prediction.",
        );
      }
    } finally {
      window.clearTimeout(timeout);
      setLoading(false);
    }
  }

  const riskClass = result?.predicted_class === 1 ? "risk-high" : "risk-low";

  return (
    <main className="app-shell">
      <header className="page-header">
        <div>
          <p className="eyebrow">AI RETENTION PLATFORM</p>

          <h1>Customer churn prediction</h1>

          <p className="subtitle">
            Predict churn risk and understand the factors influencing the model.
          </p>
        </div>

        <div className="model-status">
          <span className="status-dot" />
          Model online
        </div>
      </header>

      <div className="content-grid">
        <section className="card form-card">
          <div className="card-heading">
            <div>
              <h2>Customer information</h2>
              <p>Enter the customer attributes used by the ML model.</p>
            </div>
          </div>

          <form onSubmit={handleSubmit} className="customer-form">
            {(Object.keys(fieldLabels) as Array<keyof CustomerForm>).map(
              (field) => {
                const isNumeric = numericFields.includes(field);

                return (
                  <label className="field" key={field}>
                    <span>{fieldLabels[field]}</span>

                    {isNumeric ? (
                      <input
                        type="number"
                        min={field === "SeniorCitizen" ? 0 : 0}
                        max={
                          field === "SeniorCitizen"
                            ? 1
                            : field === "tenure"
                              ? 72
                              : undefined
                        }
                        step={
                          field === "SeniorCitizen" || field === "tenure"
                            ? 1
                            : 0.01
                        }
                        value={customer[field]}
                        onChange={(event) =>
                          updateField(field, event.target.value)
                        }
                        required
                      />
                    ) : (
                      <select
                        value={String(customer[field])}
                        onChange={(event) =>
                          updateField(field, event.target.value)
                        }
                        required
                      >
                        {selectOptions[field]?.map((option) => (
                          <option value={option} key={option}>
                            {option}
                          </option>
                        ))}
                      </select>
                    )}
                  </label>
                );
              },
            )}

            <div className="form-actions">
              <button
                type="button"
                className="button secondary"
                onClick={resetForm}
                disabled={loading}
              >
                Reset
              </button>

              <button
                type="submit"
                className="button primary"
                disabled={loading}
              >
                {loading ? "Predicting..." : "Predict churn"}
              </button>
            </div>
          </form>

          {error && (
            <div className="error-message" role="alert">
              {error}
            </div>
          )}
        </section>

        <section className="card result-card">
          {!result && !loading && (
            <div className="empty-state">
              <div className="empty-icon">AI</div>

              <h2>No prediction yet</h2>

              <p>
                Submit customer information to view the churn prediction and
                model explanation.
              </p>
            </div>
          )}

          {loading && (
            <div className="empty-state">
              <div className="loader" />
              <h2>Analyzing customer</h2>
              <p>The model is calculating churn probability.</p>
            </div>
          )}

          {result && !loading && (
            <>
              <div className="result-header">
                <div>
                  <p className="eyebrow">PREDICTION RESULT</p>

                  <h2>{result.prediction_label}</h2>
                </div>

                <span className={`risk-badge ${riskClass}`}>
                  {result.predicted_class === 1 ? "High risk" : "Lower risk"}
                </span>
              </div>

              <div className="probability-panel">
                <span>Churn probability</span>

                <strong>{(result.churn_probability * 100).toFixed(2)}%</strong>

                <div className="progress-track">
                  <div
                    className={`progress-fill ${riskClass}`}
                    style={{
                      width: `${Math.min(
                        result.churn_probability * 100,
                        100,
                      )}%`,
                    }}
                  />
                </div>
              </div>

              <div className="factors-section">
                <div className="section-heading">
                  <h3>Main influencing factors</h3>

                  <span>SHAP explanation</span>
                </div>

                <div className="factor-list">
                  {result.top_factors.map((factor) => (
                    <div
                      className="factor"
                      key={`${factor.feature}-${factor.shap_value}`}
                    >
                      <div>
                        <strong>{factor.feature}</strong>

                        <small>Current value: {String(factor.value)}</small>
                      </div>

                      <span
                        className={
                          factor.shap_value > 0 ? "factor-up" : "factor-down"
                        }
                      >
                        {factor.shap_value > 0 ? "+" : ""}
                        {factor.shap_value.toFixed(3)}
                      </span>
                    </div>
                  ))}
                </div>
              </div>
            </>
          )}
        </section>
      </div>
    </main>
  );
}

export default App;
