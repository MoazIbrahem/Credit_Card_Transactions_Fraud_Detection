"""
pages/prediction.py
Prediction page: accepts credit card behavioral features,
preprocesses them, runs selected clustering models,
and displays segment assignments with confidence scores.
"""

import os
import sys
import numpy as np
import pandas as pd
import torch
from dash import html, dcc, Input, Output, State, callback
import dash_bootstrap_components as dbc

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.append(BASE_DIR)

from components.cards import result_card, section_header, glass_card

# =============================================================================
# Load models once at import time
# =============================================================================
try:
    from utils.load_models import load_all, Autoencoder
    models, scaler, features, cluster_names, metrics = load_all()
    MODELS_LOADED = True
except Exception as e:
    print(f"⚠️  Could not load models: {e}")
    models, scaler, features, cluster_names, metrics = {}, None, [], {}, {}
    MODELS_LOADED = False

# =============================================================================
# Load autoencoder separately for GMM pre-encoding
# =============================================================================
_autoencoder = None
try:
    _ae_path = os.path.join(BASE_DIR, "saved_models", "autoencoder.pt")
    if os.path.exists(_ae_path) and features:
        _autoencoder = Autoencoder(input_dim=len(features), encoding_dim=8)
        _autoencoder.load_state_dict(torch.load(_ae_path, map_location="cpu"))
        _autoencoder.eval()
except Exception as e:
    print(f"⚠️  Could not load autoencoder for GMM: {e}")

# Models that need encoded (8-dim) input instead of raw 17-dim scaled input
ENCODED_MODELS = {"GMM"}

# =============================================================================
# Color palette per model
# =============================================================================
MODEL_COLORS = {
    "K-Means":      "#64ffda",
    "AE + K-Means": "#34c5e2",
    "DBSCAN":       "#a78bfa",
    "GMM":          "#f472b6",
    "SOM":          "#fb923c",
}

# =============================================================================
# Layout
# =============================================================================
def prediction_layout():
    model_options = [{"label": m, "value": m} for m in (models.keys() if MODELS_LOADED else [])]

    return html.Div([
        section_header(
            "Customer Segmentation",
            "Enter a customer's behavioral profile to discover their segment across models."
        ),

        dbc.Row([
            # ── Left Column: Input Form ───────────────────────────────────────
            dbc.Col([
                glass_card(
                    html.Div([
                        html.H5("Select Models", className="form-section-title"),
                        dcc.Dropdown(
                            id="pred-model-selector",
                            options=model_options,
                            multi=True,
                            placeholder="Choose one or more models…",
                            className="custom-dropdown",
                            value=list(models.keys()) if MODELS_LOADED else []
                        ),
                    ], className="mb-4"),

                    html.Hr(className="form-divider"),

                    html.H5("Account Behavior", className="form-section-title"),

                    dbc.Row([
                        dbc.Col(_num_input("pred-balance",   "Balance (£)",     "e.g. 1500.00"), width=6),
                        dbc.Col(_num_input("pred-purchases", "Total Purchases",  "e.g. 800.00"),  width=6),
                    ], className="mb-3"),

                    dbc.Row([
                        dbc.Col(_num_input("pred-oneoff-purchases",  "One-off Purchases",    "e.g. 500.00"), width=6),
                        dbc.Col(_num_input("pred-installments",      "Installment Purchases","e.g. 300.00"), width=6),
                    ], className="mb-3"),

                    dbc.Row([
                        dbc.Col(_num_input("pred-cash-advance",  "Cash Advance",  "e.g. 0.00"),    width=6),
                        dbc.Col(_num_input("pred-credit-limit",  "Credit Limit",  "e.g. 5000.00"), width=6),
                    ], className="mb-3"),

                    dbc.Row([
                        dbc.Col(_num_input("pred-payments",     "Payments Made",    "e.g. 1200.00"), width=6),
                        dbc.Col(_num_input("pred-min-payments", "Minimum Payments", "e.g. 150.00"),  width=6),
                    ], className="mb-3"),

                    html.Hr(className="form-divider"),
                    html.H5("Frequency & Ratios", className="form-section-title"),

                    dbc.Row([
                        dbc.Col(_slider_input("pred-balance-freq",   "Balance Frequency",   0, 1, 0.01, 0.8), width=6),
                        dbc.Col(_slider_input("pred-purchases-freq", "Purchases Frequency", 0, 1, 0.01, 0.5), width=6),
                    ], className="mb-3"),

                    dbc.Row([
                        dbc.Col(_slider_input("pred-oneoff-freq",  "One-off Purchase Frequency", 0, 1, 0.01, 0.3), width=6),
                        dbc.Col(_slider_input("pred-install-freq", "Installments Frequency",     0, 1, 0.01, 0.3), width=6),
                    ], className="mb-3"),

                    dbc.Row([
                        dbc.Col(_slider_input("pred-cash-advance-freq", "Cash Advance Frequency", 0, 1, 0.01, 0.0), width=6),
                        dbc.Col(_slider_input("pred-prc-full",          "% Full Payment",         0, 1, 0.01, 0.2), width=6),
                    ], className="mb-3"),

                    dbc.Row([
                        dbc.Col(_num_input("pred-cash-advance-trx", "Cash Advance Trx", "e.g. 0"),  width=6),
                        dbc.Col(_num_input("pred-purchases-trx",    "Purchases Trx",    "e.g. 10"), width=6),
                    ], className="mb-3"),

                    dbc.Row([
                        dbc.Col(_num_input("pred-tenure", "Tenure (months)", "e.g. 12"), width=6),
                    ], className="mb-4"),

                    html.Button(
                        [html.Span("◈", className="btn-icon"), " Run Segmentation"],
                        id="pred-run-btn",
                        n_clicks=0,
                        className="run-button w-100"
                    ),
                    extra_class="form-card"
                )
            ], width=5),

            # ── Right Column: Results ─────────────────────────────────────────
            dbc.Col([
                html.Div(id="pred-results-area", children=[
                    _empty_state()
                ], className="results-area")
            ], width=7),
        ], className="g-4"),
    ], className="prediction-page")


# =============================================================================
# Helper: input components
# =============================================================================
def _num_input(id_, label, placeholder):
    return html.Div([
        html.Label(label, className="input-label"),
        dcc.Input(
            id=id_,
            type="number",
            placeholder=placeholder,
            min=0,
            className="form-input",
            debounce=False,
        )
    ], className="input-group-custom")


def _slider_input(id_, label, min_, max_, step, default):
    return html.Div([
        html.Label(
            [label, html.Span(id=f"{id_}-display", className="slider-value-display")],
            className="input-label"
        ),
        dcc.Slider(
            id=id_,
            min=min_, max=max_, step=step, value=default,
            marks=None,
            tooltip={"placement": "bottom", "always_visible": False},
            className="custom-slider"
        )
    ], className="input-group-custom")


def _empty_state():
    return html.Div([
        html.Div("◈", className="empty-icon"),
        html.H4("No Prediction Yet", className="empty-title"),
        html.P("Fill in the customer profile on the left and click Run Segmentation.",
               className="empty-subtitle"),
    ], className="empty-state")


# =============================================================================
# Callback: Run Prediction
# =============================================================================
@callback(
    Output("pred-results-area", "children"),
    Input("pred-run-btn", "n_clicks"),
    [
        State("pred-model-selector",    "value"),
        State("pred-balance",           "value"),
        State("pred-purchases",         "value"),
        State("pred-oneoff-purchases",  "value"),
        State("pred-installments",      "value"),
        State("pred-cash-advance",      "value"),
        State("pred-credit-limit",      "value"),
        State("pred-payments",          "value"),
        State("pred-min-payments",      "value"),
        State("pred-balance-freq",      "value"),
        State("pred-purchases-freq",    "value"),
        State("pred-oneoff-freq",       "value"),
        State("pred-install-freq",      "value"),
        State("pred-cash-advance-freq", "value"),
        State("pred-prc-full",          "value"),
        State("pred-cash-advance-trx",  "value"),
        State("pred-purchases-trx",     "value"),
        State("pred-tenure",            "value"),
    ],
    prevent_initial_call=True
)
def run_prediction(n_clicks, selected_models,
                   balance, purchases, oneoff, installments, cash_adv,
                   credit_limit, payments, min_payments,
                   bal_freq, purch_freq, oneoff_freq, install_freq,
                   cash_adv_freq, prc_full, cash_adv_trx, purch_trx, tenure):

    # ── Validation ────────────────────────────────────────────────────────────
    if not selected_models:
        return _alert("Please select at least one model.", "warning")

    raw_values = [balance, purchases, oneoff, installments, cash_adv,
                  credit_limit, payments, min_payments,
                  bal_freq, purch_freq, oneoff_freq, install_freq,
                  cash_adv_freq, prc_full, cash_adv_trx, purch_trx, tenure]

    if any(v is None for v in raw_values):
        return _alert("Please fill in ALL fields before running segmentation.", "danger")

    if not MODELS_LOADED:
        return _alert("Models are not loaded. Please check the server logs.", "danger")

    # ── Build & scale input (17-dim) ──────────────────────────────────────────
    try:
        input_data = {
            "BALANCE":                          float(balance),
            "BALANCE_FREQUENCY":                float(bal_freq),
            "PURCHASES":                        float(purchases),
            "ONEOFF_PURCHASES":                 float(oneoff),
            "INSTALLMENTS_PURCHASES":           float(installments),
            "CASH_ADVANCE":                     float(cash_adv),
            "PURCHASES_FREQUENCY":              float(purch_freq),
            "ONEOFF_PURCHASES_FREQUENCY":       float(oneoff_freq),
            "PURCHASES_INSTALLMENTS_FREQUENCY": float(install_freq),
            "CASH_ADVANCE_FREQUENCY":           float(cash_adv_freq),
            "CASH_ADVANCE_TRX":                 float(cash_adv_trx),
            "PURCHASES_TRX":                    float(purch_trx),
            "CREDIT_LIMIT":                     float(credit_limit),
            "PAYMENTS":                         float(payments),
            "MINIMUM_PAYMENTS":                 float(min_payments),
            "PRC_FULL_PAYMENT":                 float(prc_full),
            "TENURE":                           float(tenure),
        }

        input_df     = pd.DataFrame([input_data])
        input_df     = input_df.reindex(columns=features, fill_value=0.0)
        input_scaled = scaler.transform(input_df)          # shape (1, 17)

    except Exception as e:
        return _alert(f"Preprocessing error: {e}", "danger")

    # ── Encode to 8-dim for models that need it ───────────────────────────────
    input_encoded = None
    if _autoencoder is not None and any(m in ENCODED_MODELS for m in selected_models):
        try:
            with torch.no_grad():
                tensor   = torch.tensor(input_scaled, dtype=torch.float32)
                encoded  = _autoencoder.encode(tensor).numpy()   # shape (1, 8)
            input_encoded = encoded
        except Exception as e:
            print(f"⚠️  Autoencoder encoding failed: {e}")

    # ── Run each selected model ───────────────────────────────────────────────
    results = []
    for model_name in selected_models:
        try:
            model = models[model_name]

            # Choose correct input dimensionality
            if model_name in ENCODED_MODELS:
                if input_encoded is None:
                    results.append(_alert(
                        f"{model_name} requires the autoencoder encoder, which failed to load.",
                        "danger"
                    ))
                    continue
                X_input = input_encoded          # (1, 8)
            else:
                X_input = input_scaled           # (1, 17)

            cluster_id = int(model.predict(X_input)[0])

            # Confidence score
            if hasattr(model, "predict_proba"):
                proba      = model.predict_proba(X_input)[0]
                confidence = float(np.max(proba))
            else:
                confidence = 0.75

            # Cluster name
            algo_key = {
                "K-Means":      "kmeans",
                "AE + K-Means": "ae_kmeans",
                "DBSCAN":       "dbscan",
                "GMM":          "gmm",
                "SOM":          "som",
            }.get(model_name, "kmeans")

            names_dict = cluster_names.get(algo_key, {})
            cname      = names_dict.get(cluster_id, f"Segment {cluster_id}")
            color      = MODEL_COLORS.get(model_name, "#64ffda")

            results.append(result_card(model_name, cluster_id, cname, confidence, color))

        except Exception as e:
            results.append(_alert(f"{model_name} prediction failed: {e}", "danger"))

    header = html.Div([
        html.H4("Segmentation Results", className="results-header-title"),
        html.P(f"Ran {len(selected_models)} model(s) successfully.",
               className="results-header-sub"),
    ], className="results-header")

    return [header] + results


# =============================================================================
# Helper: alert component
# =============================================================================
def _alert(message: str, color: str = "warning"):
    icon = {"warning": "⚠", "danger": "✕", "success": "✓"}.get(color, "ℹ")
    return html.Div([
        html.Span(icon, className="alert-icon"),
        html.Span(message, className="alert-message"),
    ], className=f"custom-alert alert-{color}")