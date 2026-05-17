"""
components/cards.py
Reusable card and UI components used across pages.
"""

from dash import html
import dash_bootstrap_components as dbc


def stat_card(label: str, value: str, icon: str = "◈", color_class: str = "accent-teal"):
    """A KPI / stat card with an icon, label, and value."""
    return html.Div([
        html.Div(icon, className=f"stat-card-icon {color_class}"),
        html.Div([
            html.P(label, className="stat-card-label"),
            html.H3(value, className="stat-card-value"),
        ], className="stat-card-body")
    ], className="stat-card")


def result_card(model_name: str, cluster_id: int, cluster_name: str,
                confidence: float, color: str = "#64ffda"):
    """A prediction result card for a single model."""
    confidence_pct = confidence * 100
    return html.Div([
        html.Div([
            html.Span(model_name, className="result-model-name"),
            html.Span(f"{confidence_pct:.1f}%", className="result-confidence"),
        ], className="result-card-header"),
        html.Div([
            html.Div(f"Cluster {cluster_id}", className="result-cluster-id"),
            html.H4(cluster_name, className="result-cluster-name"),
        ], className="result-cluster-block"),
        html.Div([
            html.Div(
                className="result-progress-fill",
                style={"width": f"{confidence_pct:.1f}%",
                       "backgroundColor": color}
            )
        ], className="result-progress-bar"),
        html.P(f"Assignment confidence: {confidence_pct:.1f}%",
               className="result-confidence-label"),
    ], className="result-card",
       style={"borderLeftColor": color})


def metric_badge(label: str, value: str, good: bool = True):
    """Small badge showing a metric value with good/bad coloring."""
    cls = "metric-badge good" if good else "metric-badge bad"
    return html.Div([
        html.Span(label, className="metric-badge-label"),
        html.Span(value, className="metric-badge-value"),
    ], className=cls)


def section_header(title: str, subtitle: str = ""):
    """Page section header with title and optional subtitle."""
    return html.Div([
        html.H2(title, className="section-title"),
        html.P(subtitle, className="section-subtitle") if subtitle else None,
    ], className="section-header")


def glass_card(*children, extra_class: str = ""):
    """Generic glass-morphism container card."""
    return html.Div(children, className=f"glass-card {extra_class}".strip())
