import os
import sys
import json
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from dash import html, dcc, Input, Output, State, callback
import dash_bootstrap_components as dbc

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.append(BASE_DIR)

from components.cards import section_header, glass_card


# Load metrics once at import time
METRICS_PATH = os.path.join(BASE_DIR, "metrics", "clustering_metrics.json")
try:
    with open(METRICS_PATH, "r") as f:
        METRICS = json.load(f)
except Exception as e:
    print(f"⚠️  Could not load metrics: {e}")
    # Demo / fallback data so the GUI is always renderable
    METRICS = {
        "K-Means":      {"silhouette": 0.21, "davies_bouldin": 1.42, "calinski_harabasz": 3204, "n_clusters": 4,  "noise_pct": 0.0},
        "AE + K-Means": {"silhouette": 0.31, "davies_bouldin": 1.18, "calinski_harabasz": 4571, "n_clusters": 4,  "noise_pct": 0.0},
        "DBSCAN":       {"silhouette": 0.27, "davies_bouldin": 1.63, "calinski_harabasz": 2187, "n_clusters": 3,  "noise_pct": 6.2},
        "GMM":          {"silhouette": 0.29, "davies_bouldin": 1.29, "calinski_harabasz": 3980, "n_clusters": 4,  "noise_pct": 0.0},
        "SOM":          {"silhouette": 0.24, "davies_bouldin": 1.35, "calinski_harabasz": 3410, "n_clusters": 5,  "noise_pct": 0.0},
    }

ALL_MODELS   = list(METRICS.keys())
MODEL_COLORS = {
    "K-Means":      "#64ffda",
    "AE + K-Means": "#34c5e2",
    "DBSCAN":       "#a78bfa",
    "GMM":          "#f472b6",
    "SOM":          "#fb923c",
}


# Metric definitions
METRIC_META = {
    "silhouette":          {"label": "Silhouette Score",       "higher_better": True,  "fmt": ".4f", "range": [-1, 1]},
    "davies_bouldin":      {"label": "Davies-Bouldin Index",   "higher_better": False, "fmt": ".4f", "range": [0,  5]},
    "calinski_harabasz":   {"label": "Calinski-Harabasz Index","higher_better": True,  "fmt": ",.0f","range": [0,  None]},
}


# Layout
def comparison_layout():
    options = [{"label": m, "value": m} for m in ALL_MODELS]

    return html.Div([
        #  Page Header 
        section_header(
            "Model Comparison",
            "Evaluate and compare all clustering algorithms side by side."
        ),

        #  Model Selector 
        glass_card(
            html.Div([
                html.H5("Select Models to Compare", className="form-section-title"),
                dcc.Checklist(
                    id="comp-model-checklist",
                    options=options,
                    value=ALL_MODELS,
                    inline=True,
                    className="model-checklist",
                    inputClassName="checklist-input",
                    labelClassName="checklist-label",
                ),
            ])
        ),

        html.Div(id="comp-best-banner", className="mt-3"),

        #  KPI Row 
        html.Div(id="comp-kpi-row", className="kpi-row mt-4"),

        #  Charts Row 
        dbc.Row([
            dbc.Col([
                glass_card(
                    html.H5("Silhouette Score ↑", className="chart-title"),
                    html.P("Higher is better. Range: −1 to 1.", className="chart-subtitle"),
                    dcc.Graph(id="comp-chart-silhouette", config={"displayModeBar": False}),
                )
            ], width=6),
            dbc.Col([
                glass_card(
                    html.H5("Davies-Bouldin Index ↓", className="chart-title"),
                    html.P("Lower is better. Measures intra/inter-cluster ratio.", className="chart-subtitle"),
                    dcc.Graph(id="comp-chart-dbi", config={"displayModeBar": False}),
                )
            ], width=6),
        ], className="mt-4 g-4"),

        dbc.Row([
            dbc.Col([
                glass_card(
                    html.H5("Calinski-Harabasz Index ↑", className="chart-title"),
                    html.P("Higher is better. Between-vs-within cluster dispersion ratio.", className="chart-subtitle"),
                    dcc.Graph(id="comp-chart-ch", config={"displayModeBar": False}),
                )
            ], width=8),
            dbc.Col([
                glass_card(
                    html.H5("Cluster Properties", className="chart-title"),
                    html.P("Number of clusters and noise percentage.", className="chart-subtitle"),
                    html.Div(id="comp-properties-table"),
                )
            ], width=4),
        ], className="mt-4 g-4"),

    ], className="comparison-page")



# Callback: update all comparison outputs
@callback(
    [
        Output("comp-best-banner",         "children"),
        Output("comp-kpi-row",             "children"),
        Output("comp-chart-silhouette",    "figure"),
        Output("comp-chart-dbi",           "figure"),
        Output("comp-chart-ch",            "figure"),
        Output("comp-properties-table",    "children"),
    ],
    Input("comp-model-checklist", "value"),
    prevent_initial_call=False
)
def update_comparison(selected):
    if not selected:
        empty = go.Figure()
        _style_fig(empty)
        return html.Div(), html.Div(), empty, empty, empty, html.Div()

    data = {m: METRICS[m] for m in selected if m in METRICS}
    models  = list(data.keys())
    colors  = [MODEL_COLORS[m] for m in models]

    #  Best model (composite rank) 
    best_model = _find_best(data)
    best_banner = _build_best_banner(best_model, data[best_model])

    #  KPI Cards 
    kpi_cards = _build_kpi_cards(data, best_model)

    #  Bar charts 
    fig_sil = _bar_chart(models, [data[m]["silhouette"]         for m in models], colors,
                         "Silhouette Score", higher_better=True)
    fig_dbi = _bar_chart(models, [data[m]["davies_bouldin"]     for m in models], colors,
                         "Davies-Bouldin Index", higher_better=False)
    fig_ch  = _bar_chart(models, [data[m]["calinski_harabasz"]  for m in models], colors,
                         "Calinski-Harabasz Index", higher_better=True)

    #  Properties table 
    prop_table = _properties_table(data)

    return best_banner, kpi_cards, fig_sil, fig_dbi, fig_ch, prop_table



# Helper: find best model by composite rank
def _find_best(data: dict) -> str:
    scores = {}
    for m, v in data.items():
        # Normalise each metric to [0,1] where 1 = best
        n_sil  = v["silhouette"]                  # already [-1,1], higher better
        n_dbi  = 1 / (1 + v["davies_bouldin"])    # lower better → invert
        # CH: normalise relative to max in selection
        ch_vals = [d["calinski_harabasz"] for d in data.values()]
        n_ch = v["calinski_harabasz"] / max(ch_vals) if max(ch_vals) > 0 else 0
        scores[m] = (n_sil + n_dbi + n_ch) / 3
    return max(scores, key=scores.get)



# Helper: best model banner
def _build_best_banner(model_name: str, metrics: dict) -> html.Div:
    color = MODEL_COLORS.get(model_name, "#64ffda")
    return html.Div([
        html.Div([
            html.Span("🏆", className="banner-trophy"),
            html.Div([
                html.Span("Best Overall Model", className="banner-label"),
                html.H4(model_name, className="banner-model-name", style={"color": color}),
            ]),
        ], className="banner-left"),
        html.Div([
            html.Div([html.Span("Silhouette", className="banner-metric-label"),
                      html.Span(f"{metrics['silhouette']:.4f}", className="banner-metric-value")]),
            html.Div([html.Span("Davies-Bouldin", className="banner-metric-label"),
                      html.Span(f"{metrics['davies_bouldin']:.4f}", className="banner-metric-value")]),
            html.Div([html.Span("Clusters", className="banner-metric-label"),
                      html.Span(str(metrics["n_clusters"]), className="banner-metric-value")]),
        ], className="banner-right"),
    ], className="best-banner", style={"borderColor": color})



# Helper: KPI row
def _build_kpi_cards(data: dict, best_model: str) -> html.Div:
    sil_vals = {m: v["silhouette"]          for m, v in data.items()}
    dbi_vals = {m: v["davies_bouldin"]      for m, v in data.items()}
    ch_vals  = {m: v["calinski_harabasz"]   for m, v in data.items()}

    best_sil = max(sil_vals, key=sil_vals.get)
    best_dbi = min(dbi_vals, key=dbi_vals.get)
    best_ch  = max(ch_vals,  key=ch_vals.get)

    cards = []
    for model_name, v in data.items():
        color = MODEL_COLORS.get(model_name, "#64ffda")
        is_best = model_name == best_model
        cards.append(html.Div([
            html.Div(model_name, className="kpi-model-name",
                     style={"color": color}),
            html.Div("★ Best Overall" if is_best else "", className="kpi-best-badge",
                     style={"opacity": "1" if is_best else "0"}),
            html.Div([
                _kpi_metric("Silhouette",     f"{v['silhouette']:.4f}",
                            highlight=(model_name == best_sil)),
                _kpi_metric("Davies-Bouldin", f"{v['davies_bouldin']:.4f}",
                            highlight=(model_name == best_dbi)),
                _kpi_metric("Cal-Harabasz",   f"{v['calinski_harabasz']:,.0f}",
                            highlight=(model_name == best_ch)),
                _kpi_metric("Clusters",       str(v["n_clusters"])),
            ], className="kpi-metrics-grid"),
            html.Div(className="kpi-accent-bar", style={"backgroundColor": color}),
        ], className=f"kpi-card {'kpi-card-best' if is_best else ''}",
           style={"borderTopColor": color}))

    return html.Div(cards, className="kpi-row-inner")


def _kpi_metric(label, value, highlight=False):
    cls = "kpi-metric kpi-metric-highlight" if highlight else "kpi-metric"
    return html.Div([
        html.Span(label, className="kpi-metric-label"),
        html.Span(value, className="kpi-metric-value"),
    ], className=cls)



# Helper: bar chart factory
def _bar_chart(models, values, colors, title, higher_better=True):
    # Mark best bar
    best_idx = np.argmax(values) if higher_better else np.argmin(values)

    fig = go.Figure()
    for i, (m, v, c) in enumerate(zip(models, values, colors)):
        fig.add_trace(go.Bar(
            x=[m], y=[v],
            name=m,
            marker_color=c,
            marker_opacity=1.0 if i == best_idx else 0.55,
            marker_line_width=2 if i == best_idx else 0,
            marker_line_color=c,
            text=[f"{v:.4f}" if v < 10000 else f"{v:,.0f}"],
            textposition="outside",
            textfont={"color": c if i == best_idx else "#94a3b8", "size": 12},
        ))

    _style_fig(fig, height=280)
    fig.update_layout(
        showlegend=False,
        bargap=0.3,
        yaxis={"gridcolor": "rgba(255,255,255,0.05)", "zeroline": False},
        xaxis={"tickfont": {"color": "#cbd5e1", "size": 12}},
    )
    return fig



# Helper: properties table
def _properties_table(data: dict) -> html.Div:
    rows = []
    for m, v in data.items():
        color = MODEL_COLORS.get(m, "#64ffda")
        rows.append(html.Tr([
            html.Td(html.Span("●", style={"color": color})),
            html.Td(m,               className="prop-model"),
            html.Td(v["n_clusters"], className="prop-val"),
            html.Td(f"{v['noise_pct']:.1f}%", className="prop-val"),
        ]))

    return html.Table([
        html.Thead(html.Tr([
            html.Th(""),
            html.Th("Algorithm"),
            html.Th("Clusters"),
            html.Th("Noise %"),
        ], className="prop-thead")),
        html.Tbody(rows),
    ], className="prop-table")



# Shared figure styling
def _style_fig(fig: go.Figure, height: int = 300):
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"family": "DM Mono, monospace", "color": "#cbd5e1"},
        height=height,
        margin={"l": 20, "r": 20, "t": 20, "b": 20},
    )