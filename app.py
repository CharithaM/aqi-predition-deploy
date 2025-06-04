import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State
import plotly.express as px
import pandas as pd
import dash_bootstrap_components as dbc
import datetime
import numpy as np

# Initialize Dash app with Bootstrap theme
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.SUPERHERO])

# Background gradient styling
gradient_background = {
    "background": "linear-gradient(to right, #4A00E0, #8E2DE2, #FF0080)",
    "minHeight": "100vh",
    "padding": "20px",
    "overflowX": "hidden",
    "backgroundAttachment": "fixed",
    "backgroundRepeat": "no-repeat",
    "backgroundSize": "cover"
}

# App layout
app.layout = html.Div(style=gradient_background, children=[
    dbc.Container([
        dbc.Row([
            dbc.Col(
                html.H1("Monthly AQI Prediction Dashboard", className="text-center text-white mb-4"),
                width=12
            )
        ]),

        dbc.Row([
            dbc.Col([
                html.Label("Select Date Range:", className="fw-bold text-white"),
                dcc.DatePickerRange(
                    id='date-picker',
                    start_date=datetime.date(2025, 6, 4),
                    end_date=datetime.date(2025, 6, 4),
                    min_date_allowed=datetime.date(2015, 7, 1),
                    max_date_allowed=datetime.date(2030, 12, 31),
                    display_format='YYYY-MM-DD',
                    className="rounded p-2",
                    style={
                        "borderRadius": "12px",
                        "backgroundColor": "#1e1e1e",
                        "color": "white",
                        "border": "1px solid #ccc",
                        "padding": "8px"
                    }
                )
            ], md=6, lg=4, className="mb-2"),

            dbc.Col([
                html.Label(" ", className="d-block"),
                dbc.Button("AQI Threshold", id="open-threshold-modal", color="info",
                           style={"borderRadius": "12px", "width": "100%"})
            ], md=3, lg=2, className="mb-2"),

            dbc.Col([
                html.Label(" ", className="d-block"),
                dbc.Button("Predict", id="predict-button", color="success",
                           style={"borderRadius": "12px", "width": "100%"})
            ], md=3, lg=2, className="mb-2")
        ], className="justify-content-center g-3 mb-4"),

        dbc.Modal([
            dbc.ModalHeader(dbc.ModalTitle("AQI Threshold Levels")),
            dbc.ModalBody(
                html.Div([
                    html.Ul([
                        html.Li("0–50: Good", style={"color": "#009966", "fontSize": "16px"}),
                        html.Li("51–100: Satisfactory", style={"color": "#ffde33", "fontSize": "16px"}),
                        html.Li("101–200: Moderate", style={"color": "#ff9933", "fontSize": "16px"}),
                        html.Li("201–300: Poor", style={"color": "#cc0033", "fontSize": "16px"}),
                        html.Li("301–400: Very Poor", style={"color": "#660099", "fontSize": "16px"}),
                        html.Li(">= 400: Severe", style={"color": "#7e0023", "fontSize": "16px"})
                    ])
                ], style={
                    "backgroundColor": "white",
                    "padding": "20px",
                    "borderRadius": "10px"
                })
            ),
            dbc.ModalFooter(
                dbc.Button("Close", id="close-threshold-modal", className="ms-auto", n_clicks=0)
            )
        ],
            id="threshold-modal",
            is_open=False,
            size="lg",
            backdrop=True,
            centered=True,
            scrollable=True
        ),

        dbc.Row([
            dbc.Col(
                dbc.Card(
                    dbc.CardBody(html.Div(id="graph-container")),
                    className="shadow-lg bg-dark text-white rounded"
                ),
                width=12
            )
        ]),

        dbc.Row([
            dbc.Col(
                html.Div(id="status-message", className="text-center mt-3 text-white"),
                width=12
            )
        ])
    ], fluid=True)
])

# Load and preprocess data
def get_lstm_data():
    df = pd.read_csv('lstm_preds.csv', parse_dates=['Date'])
    df['Month'] = df['Date'].dt.to_period('M')
    monthly_df = df.groupby('Month').first().reset_index()
    monthly_df['Date'] = monthly_df['Month'].dt.to_timestamp()
    return monthly_df[['Date', 'LSTM_Prediction']]

prediction_data = get_lstm_data()

# Callback to render prediction graph
@app.callback(
    [Output('graph-container', 'children'),
     Output('status-message', 'children')],
    [Input('predict-button', 'n_clicks')],
    [State('date-picker', 'start_date'),
     State('date-picker', 'end_date')]
)
def update_graph(n_clicks, start_date, end_date):
    if not n_clicks:
        fig = px.line(title="Select date range and click Predict", template="plotly_dark")
        fig.update_layout(
            xaxis=dict(showgrid=False),
            yaxis=dict(showgrid=False),
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)'
        )
        return [dcc.Graph(figure=fig)], ""
    
    try:
        start_date = pd.to_datetime(start_date)
        end_date = pd.to_datetime(end_date)

        start_month = pd.to_datetime(start_date).to_period('M')
        end_month = pd.to_datetime(end_date).to_period('M')

        if start_month == end_month:
            date_range = [start_month]
        else:
            date_range = pd.period_range(start=start_month, end=end_month, freq='M')

        filtered_df = prediction_data[prediction_data['Date'].dt.to_period('M').isin(date_range)]

        if filtered_df.empty:
            return [], html.Div("No predictions available for selected period. Try a broader date range (2015–2025).", className="text-warning")

        fig = px.line(
            filtered_df,
            x='Date',
            y='LSTM_Prediction',
            title="Monthly AQI Predictions",
            template="plotly_dark",
            markers=True
        )

        fig.update_layout(
            hovermode="x unified",
            xaxis_title="Month",
            yaxis_title="AQI Prediction",
            showlegend=False
        )

        fig.update_traces(
            hovertemplate="<b>%{x|%B %Y}</b><br>AQI: %{y:.1f}",
            line=dict(width=3),
            marker=dict(size=8, color='white', line=dict(width=2, color='DarkSlateGrey'))
        )

        fig.update_xaxes(
            tickformat="%b %Y",
            tickangle=45,
            ticklabelmode="instant",
            ticks="outside",
            showgrid=True
        )

        aqi_ranges = {
            'Good': (0, 50),
            'Satisfactory': (51, 100),
            'Moderate': (101, 200),
            'Poor': (201, 300),
            'Very Poor': (301, 400),
            'Severe': (401, 1000)
        }

        aqi_colors = {
            'Good': '#00FF00',
            'Satisfactory': '#FFFF00',
            'Moderate': '#FF8C00',
            'Poor': '#FF0000',
            'Very Poor': '#9933FF',
            'Severe': '#FF1493'
        }

        for range_name, (min_val, max_val) in aqi_ranges.items():
            fig.add_hrect(
                y0=min_val,
                y1=max_val,
                line_width=0,
                fillcolor=aqi_colors[range_name],
                opacity=0.15,
                layer="below"
            )

        for _, row in filtered_df.iterrows():
            aqi_val = row['LSTM_Prediction']
            label_text = None
            for label, (low, high) in aqi_ranges.items():
                if low <= aqi_val <= high:
                    label_text = label
                    break
            if label_text:
                fig.add_annotation(
                    x=row['Date'],
                    y=row['LSTM_Prediction'] + 10,
                    text=label_text,
                    showarrow=False,
                    font=dict(size=12, color=aqi_colors[label_text], family="Arial Black"),
                    bgcolor="rgba(0,0,0,0.5)",
                    bordercolor=aqi_colors[label_text],
                    borderwidth=1,
                    borderpad=4,
                    opacity=0.9
                )

        return [dcc.Graph(figure=fig)], ""

    except Exception as e:
        return [], html.Div(f"Error: {str(e)}", className="text-danger")

# Toggle AQI threshold modal
@app.callback(
    Output("threshold-modal", "is_open"),
    [Input("open-threshold-modal", "n_clicks"),
     Input("close-threshold-modal", "n_clicks")],
    [State("threshold-modal", "is_open")]
)
def toggle_threshold_modal(open_clicks, close_clicks, is_open):
    if open_clicks or close_clicks:
        return not is_open
    return is_open

# Run server
if __name__ == '__main__':
    app.run_server(debug=True)
