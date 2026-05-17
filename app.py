import dash
from dash import dcc, html, Input, Output, callback_context
import dash_bootstrap_components as dbc

# App Initialization
app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.DARKLY],
    suppress_callback_exceptions=True,
    title="CreditSeg AI"
)
server = app.server  # For deployment (gunicorn)

# Import page layouts (after app is defined)
from pages.prediction import prediction_layout
from pages.comparison import comparison_layout

# App Layout
app.layout = html.Div([
    # Sidebar Navigation
    html.Div([
        html.Div([
            # Logo
            html.Div([
                html.Div("◈", className="logo-icon"),
                html.Span("CreditSeg", className="logo-text"),
            ], className="logo-container"),

            # Nav Links
            html.Nav([
                html.A(
                    [html.Span("⬡", className="nav-icon"), html.Span("Prediction", className="nav-label")],
                    id="link-pred", className="nav-link-item active", n_clicks=0
                ),
                html.A(
                    [html.Span("⬡", className="nav-icon"), html.Span("Comparison", className="nav-label")],
                    id="link-comp", className="nav-link-item", n_clicks=0
                ),
            ], className="nav-menu"),

            # Footer tag
            html.Div([
                html.Span("Customer Segmentation", className="sidebar-tag"),
                html.Span("ML Dashboard", className="sidebar-tag"),
            ], className="sidebar-footer"),
        ], className="sidebar-inner")
    ], className="sidebar"),

    # Main Content Area
    html.Div([
        html.Div(id="page-content", className="page-content")
    ], className="main-area"),

], className="app-root")

# Navigation Callbacks
@app.callback(
    [Output("link-pred", "className"),
     Output("link-comp", "className")],
    [Input("link-pred", "n_clicks"),
     Input("link-comp", "n_clicks")]
)
def update_nav_style(p, c):
    ctx = callback_context
    clicked = ctx.triggered[0]["prop_id"].split(".")[0] if ctx.triggered else "link-pred"
    base = "nav-link-item"
    return [
        f"{base} active" if clicked == "link-pred" else base,
        f"{base} active" if clicked == "link-comp" else base,
    ]


@app.callback(
    Output("page-content", "children"),
    [Input("link-pred", "n_clicks"),
     Input("link-comp", "n_clicks")]
)
def render_page(p, c):
    ctx = callback_context
    page = ctx.triggered[0]["prop_id"].split(".")[0] if ctx.triggered else "link-pred"
    if page == "link-comp":
        return comparison_layout()
    return prediction_layout()

# Run
if __name__ == "__main__":
    app.run(debug=True)
