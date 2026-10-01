import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import App from "./App";

describe("Customer Churn Prediction page", () => {
  it("renders the page title", () => {
    render(<App />);

    expect(
      screen.getByRole("heading", {
        name: /customer churn prediction/i,
      }),
    ).toBeInTheDocument();
  });

  it("renders the prediction button", () => {
    render(<App />);

    expect(
      screen.getByRole("button", {
        name: /predict churn/i,
      }),
    ).toBeInTheDocument();
  });
});
