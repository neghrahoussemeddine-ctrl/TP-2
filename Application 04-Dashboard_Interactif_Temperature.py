# -*- coding: utf-8 -*-
"""
APPLICATION 1.1 - TABLEAU DE BORD INTERACTIF AVEC PLOTLY
V03 : KPI + filtres temporels + detection des valeurs atypiques.

Cette version utilise Dash pour rendre les filtres et les KPI
dynamiques dans le navigateur.

Donnees synthetiques : temperature quotidienne de l'eau.
Bibliotheques : NumPy, Pandas, Plotly et Dash.
"""

from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from dash import Dash, dcc, html, dash_table, Input, Output


# ============================================================
# 1. PARAMETRES
# ============================================================
RANDOM_SEED = 42
START_DATE = "2026-01-01"
NUMBER_OF_DAYS = 90
MEAN_TEMPERATURE = 18.0
TEMPERATURE_STD = 1.5
MOVING_AVERAGE_WINDOW = 7
REFERENCE_TEMPERATURE = 20.0

APP_TITLE = "TABLEAU DE BORD INTERACTIF - TEMPERATURE DE L'EAU"


# ============================================================
# 2. CREATION DES DONNEES SYNTHETIQUES
# ============================================================
def create_dataset():
    """Genere une serie quotidienne reproductible avec quelques anomalies."""
    rng = np.random.default_rng(RANDOM_SEED)

    dates = pd.date_range(
        start=START_DATE,
        periods=NUMBER_OF_DAYS,
        freq="D"
    )

    # Signal de base avec une faible variation progressive.
    trend = np.linspace(0, 1.2, NUMBER_OF_DAYS)
    seasonal = 0.8 * np.sin(np.linspace(0, 3 * np.pi, NUMBER_OF_DAYS))
    temperature = (
        MEAN_TEMPERATURE
        + trend
        + seasonal
        + rng.normal(0, TEMPERATURE_STD, NUMBER_OF_DAYS)
    )

    # Quelques valeurs volontairement atypiques pour illustrer la detection.
    if NUMBER_OF_DAYS >= 60:
        temperature[18] += 5.5
        temperature[53] -= 5.0
        temperature[76] += 6.0

    data = pd.DataFrame({
        "Date": dates,
        "Temperature": temperature
    })
    return data


# ============================================================
# 3. VALIDATION ET DETECTION DES VALEURS ATYPIQUES
# ============================================================
def validate_dataset(data):
    required = {"Date", "Temperature"}
    missing = required.difference(data.columns)

    if missing:
        raise ValueError("Colonnes manquantes : " + ", ".join(sorted(missing)))
    if data.empty:
        raise ValueError("Le jeu de donnees est vide.")
    if data["Temperature"].isna().any():
        raise ValueError("La temperature contient des valeurs manquantes.")
    if not np.isfinite(data["Temperature"]).all():
        raise ValueError("La temperature contient des valeurs non finies.")


def detect_outliers(data, method="IQR"):
    """
    Retourne une copie du DataFrame avec :
    - Outlier : True si observation atypique ;
    - Lower_Bound / Upper_Bound : bornes de detection.

    IQR : Q1 - 1.5*IQR et Q3 + 1.5*IQR.
    Z-score : valeur absolue du score standardise > 3.
    """
    result = data.copy()
    values = result["Temperature"].astype(float)

    if method == "IQR":
        q1 = values.quantile(0.25)
        q3 = values.quantile(0.75)
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        mask = (values < lower) | (values > upper)
        method_label = "IQR (1.5 × écart interquartile)"
    else:
        mean = values.mean()
        std = values.std(ddof=1)
        if std == 0 or pd.isna(std):
            z_scores = pd.Series(0.0, index=values.index)
        else:
            z_scores = (values - mean) / std
        mask = z_scores.abs() > 3
        lower = mean - 3 * std
        upper = mean + 3 * std
        method_label = "Z-score (|z| > 3)"

    result["Outlier"] = mask
    result["Lower_Bound"] = lower
    result["Upper_Bound"] = upper
    result.attrs["method_label"] = method_label
    return result


# ============================================================
# 4. KPI
# ============================================================
def calculate_kpis(data):
    values = data["Temperature"].astype(float)
    if values.empty:
        return {
            "count": 0, "mean": np.nan, "median": np.nan,
            "minimum": np.nan, "maximum": np.nan, "std": np.nan
        }

    return {
        "count": int(values.count()),
        "mean": float(values.mean()),
        "median": float(values.median()),
        "minimum": float(values.min()),
        "maximum": float(values.max()),
        "std": float(values.std(ddof=1)) if len(values) > 1 else 0.0
    }


def kpi_card(label, value, unit="", accent="#2471A3"):
    return html.Div(
        [
            html.Div(label, style={
                "fontSize": "13px", "color": "#526273",
                "fontWeight": "bold", "marginBottom": "8px"
            }),
            html.Div(f"{value}{unit}", style={
                "fontSize": "27px", "fontWeight": "bold",
                "color": accent, "fontFamily": "Times New Roman"
            })
        ],
        style={
            "backgroundColor": "white",
            "border": "1px solid #D5DDE7",
            "borderLeft": f"5px solid {accent}",
            "borderRadius": "8px",
            "padding": "15px 17px",
            "minWidth": "145px",
            "flex": "1",
            "boxShadow": "0 2px 5px rgba(0,0,0,0.05)"
        }
    )


# ============================================================
# 5. FIGURES
# ============================================================
def make_empty_figure(message):
    fig = go.Figure()
    fig.add_annotation(
        text=message, x=0.5, y=0.5, xref="paper", yref="paper",
        showarrow=False, font=dict(size=16, color="#526273")
    )
    fig.update_layout(template="plotly_white", height=350)
    return fig


def build_figures(filtered_data, method):
    if filtered_data.empty:
        empty = make_empty_figure("Aucune observation pour cette période.")
        return empty, empty, empty, empty

    analysed = detect_outliers(filtered_data, method)
    normal = analysed.loc[~analysed["Outlier"]]
    anomalies = analysed.loc[analysed["Outlier"]]

    # Figure 1 : serie temporelle et anomalies
    fig_time = go.Figure()
    fig_time.add_trace(go.Scatter(
        x=normal["Date"], y=normal["Temperature"],
        mode="lines+markers", name="Observations normales",
        line=dict(color="#2471A3", width=2.5),
        marker=dict(size=7),
        hovertemplate="%{x|%d/%m/%Y}<br>Temperature : %{y:.2f} °C<extra></extra>"
    ))
    if not anomalies.empty:
        fig_time.add_trace(go.Scatter(
            x=anomalies["Date"], y=anomalies["Temperature"],
            mode="markers", name="Valeurs atypiques",
            marker=dict(color="#C0392B", size=13, symbol="diamond",
                        line=dict(color="white", width=1)),
            hovertemplate="<b>VALEUR ATYPIQUE</b><br>%{x|%d/%m/%Y}<br>"
                          "Temperature : %{y:.2f} °C<extra></extra>"
        ))
    fig_time.add_hline(
        y=REFERENCE_TEMPERATURE, line_dash="dash",
        line_color="#D97706", annotation_text=f"Reference : {REFERENCE_TEMPERATURE:.1f} °C"
    )
    fig_time.update_layout(
        title="Evolution de la temperature et valeurs atypiques",
        xaxis_title="Date", yaxis_title="Temperature (°C)",
        template="plotly_white", hovermode="x unified", height=390,
        legend=dict(orientation="h", y=1.12)
    )

    # Figure 2 : histogramme
    fig_hist = go.Figure()
    fig_hist.add_trace(go.Histogram(
        x=filtered_data["Temperature"], nbinsx=15,
        marker_color="#55A6A6", marker_line_color="white",
        marker_line_width=1, name="Frequence",
        hovertemplate="Temperature : %{x:.2f} °C<br>Effectif : %{y}<extra></extra>"
    ))
    mean_value = filtered_data["Temperature"].mean()
    fig_hist.add_vline(
        x=mean_value, line_dash="dash", line_color="#C0392B",
        annotation_text=f"Moyenne : {mean_value:.2f} °C"
    )
    fig_hist.update_layout(
        title="Distribution des temperatures",
        xaxis_title="Temperature (°C)", yaxis_title="Effectif",
        template="plotly_white", height=350
    )

    # Figure 3 : boxplot
    fig_box = go.Figure()
    fig_box.add_trace(go.Box(
        y=filtered_data["Temperature"], name="Temperature",
        boxpoints="outliers", marker_color="#7551A1",
        line_color="#563D7C", fillcolor="rgba(117,81,161,0.25)",
        boxmean=True
    ))
    fig_box.update_layout(
        title="Dispersion et valeurs atypiques (boxplot)",
        yaxis_title="Temperature (°C)",
        template="plotly_white", height=350
    )

    # Figure 4 : temperature et moyenne mobile
    smooth = filtered_data.sort_values("Date").copy()
    smooth["Moyenne_mobile"] = smooth["Temperature"].rolling(
        window=MOVING_AVERAGE_WINDOW, min_periods=1
    ).mean()

    fig_ma = go.Figure()
    fig_ma.add_trace(go.Scatter(
        x=smooth["Date"], y=smooth["Temperature"],
        mode="markers", name="Temperature quotidienne",
        marker=dict(color="#A7C7E7", size=7),
        hovertemplate="%{x|%d/%m/%Y}<br>%{y:.2f} °C<extra></extra>"
    ))
    fig_ma.add_trace(go.Scatter(
        x=smooth["Date"], y=smooth["Moyenne_mobile"],
        mode="lines", name=f"Moyenne mobile ({MOVING_AVERAGE_WINDOW} jours)",
        line=dict(color="#C0392B", width=3)
    ))
    fig_ma.update_layout(
        title="Tendance : temperature et moyenne mobile",
        xaxis_title="Date", yaxis_title="Temperature (°C)",
        template="plotly_white", height=350,
        hovermode="x unified"
    )

    return fig_time, fig_hist, fig_box, fig_ma


# ============================================================
# 6. DONNEES ET APPLICATION DASH
# ============================================================
df = create_dataset()
validate_dataset(df)

app = Dash(__name__)
app.title = "Dashboard Temperature - V03"

min_date = df["Date"].min().date()
max_date = df["Date"].max().date()

CARD_STYLE = {
    "backgroundColor": "white", "padding": "12px 15px",
    "border": "1px solid #D5DDE7", "borderRadius": "8px",
    "marginBottom": "12px"
}

GRAPH_STYLE = {
    "backgroundColor": "white", "padding": "5px",
    "border": "1px solid #D5DDE7", "borderRadius": "8px",
    "boxShadow": "0 2px 5px rgba(0,0,0,0.04)"
}

app.layout = html.Div(
    [
        # En-tete
        html.Div(
            [
                html.H1(
                    "TABLEAU DE BORD INTERACTIF",
                    style={"margin": "0", "fontFamily": "Times New Roman",
                           "fontSize": "30px", "fontWeight": "bold"}
                ),
                html.H3(
                    "Evolution de la temperature de l'eau | V03",
                    style={"margin": "5px 0 0 0", "fontWeight": "normal",
                           "fontSize": "17px"}
                )
            ],
            style={
                "backgroundColor": "#17365D", "color": "white",
                "padding": "22px 28px", "marginBottom": "15px"
            }
        ),

        # Filtres
        html.Div(
            [
                html.Div([
                    html.Label("Periode d'analyse", style={"fontWeight": "bold"}),
                    dcc.DatePickerRange(
                        id="date-filter",
                        min_date_allowed=min_date,
                        max_date_allowed=max_date,
                        start_date=min_date,
                        end_date=max_date,
                        display_format="DD/MM/YYYY",
                        minimum_nights=0,
                        style={"display": "block", "marginTop": "7px"}
                    )
                ], style={"flex": "2", "minWidth": "260px"}),

                html.Div([
                    html.Label("Methode de detection", style={"fontWeight": "bold"}),
                    dcc.Dropdown(
                        id="method-filter",
                        options=[
                            {"label": "IQR (1.5 × ecart interquartile)", "value": "IQR"},
                            {"label": "Z-score (|z| > 3)", "value": "Z-score"}
                        ],
                        value="IQR", clearable=False,
                        style={"marginTop": "7px", "color": "#263238"}
                    )
                ], style={"flex": "2", "minWidth": "250px"}),

                html.Div([
                    html.Label("Affichage des anomalies", style={"fontWeight": "bold"}),
                    dcc.RadioItems(
                        id="outlier-filter",
                        options=[
                            {"label": "Toutes", "value": "all"},
                            {"label": "Masquer les atypiques", "value": "hide"},
                            {"label": "Atypiques seulement", "value": "only"}
                        ],
                        value="all", inline=False,
                        style={"marginTop": "7px", "lineHeight": "1.8"}
                    )
                ], style={"flex": "2", "minWidth": "230px"})
            ],
            style={
                **CARD_STYLE, "display": "flex", "gap": "25px",
                "alignItems": "flex-start", "flexWrap": "wrap"
            }
        ),

        html.H3("INDICATEURS CLES DE PERFORMANCE (KPI)",
                style={"color": "#17365D", "fontSize": "17px"}),

        html.Div(id="kpi-container", style={
            "display": "flex", "gap": "12px", "flexWrap": "wrap",
            "marginBottom": "15px"
        }),

        # Graphiques
        html.Div(
            [
                dcc.Graph(id="time-chart", config={"displaylogo": False}),
                dcc.Graph(id="hist-chart", config={"displaylogo": False}),
                dcc.Graph(id="box-chart", config={"displaylogo": False}),
                dcc.Graph(id="ma-chart", config={"displaylogo": False})
            ],
            style={
                "display": "grid",
                "gridTemplateColumns": "repeat(2, minmax(0, 1fr))",
                "gap": "12px"
            }
        ),

        # Tableau des anomalies
        html.Div(
            [
                html.H3("OBSERVATIONS IDENTIFIEES COMME ATYPIQUES",
                        style={"color": "#17365D", "fontSize": "17px"}),
                html.Div(id="outlier-summary",
                         style={"marginBottom": "8px", "color": "#526273"}),
                dash_table.DataTable(
                    id="outlier-table",
                    columns=[
                        {"name": "Date", "id": "Date"},
                        {"name": "Temperature (°C)", "id": "Temperature"},
                        {"name": "Methode", "id": "Methode"}
                    ],
                    data=[],
                    page_size=8,
                    sort_action="native",
                    style_table={"overflowX": "auto"},
                    style_header={
                        "backgroundColor": "#17365D", "color": "white",
                        "fontWeight": "bold", "textAlign": "center"
                    },
                    style_cell={
                        "padding": "9px", "textAlign": "center",
                        "fontFamily": "Arial", "fontSize": "13px"
                    },
                    style_data_conditional=[
                        {"if": {"row_index": "odd"},
                         "backgroundColor": "#F2F5F9"}
                    ]
                )
            ],
            style={**CARD_STYLE, "marginTop": "15px"}
        ),

        html.Div(
            "Application pedagogique | Donnees synthetiques | NumPy + Pandas + Plotly + Dash",
            style={"textAlign": "center", "color": "#667788",
                   "fontSize": "12px", "padding": "12px"}
        )
    ],
    style={
        "backgroundColor": "#F2F5F9", "minHeight": "100vh",
        "paddingBottom": "20px",
        "fontFamily": "Arial, sans-serif", "color": "#263238"
    }
)


# ============================================================
# 7. CALLBACK : FILTRES -> KPI + GRAPHIQUES + TABLE
# ============================================================
@app.callback(
    Output("kpi-container", "children"),
    Output("time-chart", "figure"),
    Output("hist-chart", "figure"),
    Output("box-chart", "figure"),
    Output("ma-chart", "figure"),
    Output("outlier-table", "data"),
    Output("outlier-summary", "children"),
    Input("date-filter", "start_date"),
    Input("date-filter", "end_date"),
    Input("method-filter", "value"),
    Input("outlier-filter", "value")
)
def update_dashboard(start_date, end_date, method, outlier_display):
    filtered = df.copy()

    if start_date:
        filtered = filtered[filtered["Date"] >= pd.to_datetime(start_date)]
    if end_date:
        filtered = filtered[filtered["Date"] <= pd.to_datetime(end_date)]

    if filtered.empty:
        empty = make_empty_figure("Aucune donnee dans la periode selectionnee.")
        kpis = [kpi_card("Observations", "0", accent="#C0392B")]
        return kpis, empty, empty, empty, empty, [], "Aucune observation a analyser."

    analysed = detect_outliers(filtered, method)
    outliers = analysed.loc[analysed["Outlier"]].copy()
    kpis_data = calculate_kpis(filtered)

    # KPI calcules sur les observations de la periode choisie.
    kpis = [
        kpi_card("Nombre d'observations", f"{kpis_data['count']}", accent="#2471A3"),
        kpi_card("Temperature moyenne", f"{kpis_data['mean']:.2f}", " °C", "#238B57"),
        kpi_card("Temperature mediane", f"{kpis_data['median']:.2f}", " °C", "#7551A1"),
        kpi_card("Temperature minimale", f"{kpis_data['minimum']:.2f}", " °C", "#2471A3"),
        kpi_card("Temperature maximale", f"{kpis_data['maximum']:.2f}", " °C", "#C0392B"),
        kpi_card("Ecart-type", f"{kpis_data['std']:.2f}", " °C", "#D97706"),
        kpi_card("Valeurs atypiques", f"{len(outliers)}", accent="#C0392B")
    ]

    # Selon le filtre d'affichage, les graphiques utilisent toutes les donnees,
    # uniquement les observations normales ou uniquement les atypiques.
    if outlier_display == "hide":
        chart_data = analysed.loc[~analysed["Outlier"]].copy()
    elif outlier_display == "only":
        chart_data = outliers.copy()
    else:
        chart_data = filtered.copy()

    if chart_data.empty:
        fig_time = make_empty_figure("Aucune observation selon le filtre choisi.")
        fig_hist = make_empty_figure("Aucune observation selon le filtre choisi.")
        fig_box = make_empty_figure("Aucune observation selon le filtre choisi.")
        fig_ma = make_empty_figure("Aucune observation selon le filtre choisi.")
    else:
        fig_time, fig_hist, fig_box, fig_ma = build_figures(chart_data, method)

        # La figure temporelle marque les anomalies detectees sur la periode.
        # En mode 'hide'/'only', le filtre est deja applique aux donnees.
        if outlier_display == "all":
            # Ajouter les anomalies detectees dans la periode selectionnee.
            if not outliers.empty:
                fig_time.add_trace(go.Scatter(
                    x=outliers["Date"], y=outliers["Temperature"],
                    mode="markers", name="Atypiques detectees",
                    marker=dict(color="#C0392B", size=14, symbol="x")
                ))

    table_data = [
        {
            "Date": row["Date"].strftime("%d/%m/%Y"),
            "Temperature": round(float(row["Temperature"]), 3),
            "Methode": method
        }
        for _, row in outliers.iterrows()
    ]

    method_text = (
        "IQR : valeurs inferieures a Q1 - 1.5 × IQR ou superieures a Q3 + 1.5 × IQR."
        if method == "IQR"
        else "Z-score : valeurs dont |z| est strictement superieur a 3."
    )
    summary = (
        f"{len(outliers)} valeur(s) atypique(s) detectee(s) sur "
        f"{len(filtered)} observation(s). {method_text}"
    )

    return kpis, fig_time, fig_hist, fig_box, fig_ma, table_data, summary


# ============================================================
# 8. LANCEMENT DU SERVEUR LOCAL
# ============================================================
if __name__ == "__main__":
    print("=" * 68)
    print("DASHBOARD INTERACTIF V03")
    print("Ouvrez votre navigateur a l'adresse : http://127.0.0.1:8050")
    print("Pour arreter le serveur : Ctrl + C dans la console.")
    print("=" * 68)

    app.run(debug=False, host="127.0.0.1", port=8050)
