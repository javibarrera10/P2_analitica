import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import pandas as pd
import io
import base64
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import tensorflow as tf
import keras
import pandas as pd
import psycopg2
import plotly.express as px
import sklearn
from sklearn.preprocessing import StandardScaler
engine = psycopg2.connect(
    dbname="proyecto",
    user="postgres",
    password="Proyecto2",
    host="camilos.c9u0ykegym0m.us-east-1.rds.amazonaws.com",
    port="5432"
)
cursor = engine.cursor()

query = """
select * from data;
"""
cursor.execute(query)
result = cursor.fetchall()
data = pd.DataFrame(result, columns=['ID','LIMIT_BAL','SEX','EDUCATION','MARRIAGE','AGE','PAY_0','PAY_2','PAY_3','PAY_4','PAY_5','PAY_6','BILL_AMT1','BILL_AMT2','BILL_AMT3','BILL_AMT4','BILL_AMT5','BILL_AMT6','PAY_AMT1','PAY_AMT2','PAY_AMT3','PAY_AMT4','PAY_AMT5','PAY_AMT6','default payment next month'])
#print(data)
# Carga el modelo entrenado (reemplaza 'model.pkl' con la ruta de tu modelo guardado)
model = keras.models.load_model('C:\\Users\\camil\\Downloads\\Analitica computacional\\Proyecto_2\\Modelo_final1.keras')

# Preparar datos
#data = pd.read_excel('C:\\Users\\camil\\Downloads\\Analitica computacional\\Proyecto_2\\finales.xlsx')

# Lista de variables para el histograma y dropdown
histogram_vars = ['LIMIT_BAL', 'SEX', 'EDUCATION', 'MARRIAGE', 'AGE', 
                  'PAY_0', 'PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6']

# Inicializar la aplicación Dash
app = dash.Dash(__name__)

# Layout de la aplicación Dash
app.layout = html.Div([
    html.H1("Dashboard Interactivo de Correlaciones, Frecuencias y Predicción de Crédito"),
    html.Div([
        html.Div([
            html.Div([
            dcc.Checklist(id='var-select',
                options=[{'label': i, 'value': i} for i in data.columns.tolist()],
                value=['AGE', 'LIMIT_BAL'],  # Valores predeterminados
                labelStyle={'display': 'inline-block'}
        ),],style={'margin': '10px'}),
        ], style={'margin': '20px'}),
        #], style={'flex': '1'}),  

    ]),
    html.Div(id='correlation-heatmap'),
    html.H2("Histograma de Frecuencias"),
    dcc.Dropdown(
        id='histogram-var-select',
        options=[{'label': var, 'value': var} for var in histogram_vars],
        value='AGE'  # Valor predeterminado
    ),
    dcc.RangeSlider(
        id='age-range-slider',
        min=data['AGE'].min(),
        max=data['AGE'].max(),
        value=[data['AGE'].min(), data['AGE'].max()],
        marks={str(age): str(age) for age in range(data['AGE'].min(), data['AGE'].max(), 5)}
    ),
    html.Div(id='frequency-histogram'),
    html.Div([
    html.H2("Análisis de Incumplimiento de Pago"),
    dcc.Dropdown(
        id='category-dropdown',
        options=[
            {'label': 'Género', 'value': 'SEX'},
            {'label': 'Educación', 'value': 'EDUCATION'},
            {'label': 'Estado Civil', 'value': 'MARRIAGE'}
        ],
        value='SEX'  # Valor predeterminado
    ),
    dcc.Dropdown(
        id='value-dropdown',  # Valores se llenarán basados en la selección anterior
    ),
    html.Div(id='pie-chart-container')
    ]),
    html.H2("Predicción de Crédito"),
    # Entradas para las variables de la predicción
    html.Div([
        html.Div([
            html.Div([
                html.Label("LIMIT_BAL:", style={'padding': '5px'}),
                dcc.Input(id='input-limit_bal', type='number', value=50000, style={'width': '100%'}),],style={'margin': '10px'}),
            html.Div([
                html.Label("AGE:", style={'padding': '5px'}),
                dcc.Input(id='input-age', type='number', value=30, style={'width': '100%'}),],style={'margin': '10px'}),
            html.Div([
                html.Label("SEX:", style={'padding': '5px'}),
                dcc.Dropdown(id='input-sex', 
                    options=[
                    {'label': 'Masculino', 'value': 1},
                    {'label': 'Femenino', 'value': 2}
                ], value=1, style={'width': '100%'}),],style={'margin': '10px'}),
            html.Div([
                html.Label("EDUCATION:", style={'padding': '5px'}),
                dcc.Dropdown(id='input-education',
                    options=[
                        {'label': 'Postgrado', 'value': 1},
                        {'label': 'Universitario', 'value': 2},
                        {'label': 'Secundaria', 'value': 3},
                        {'label': 'Otros', 'value': 4}
                    ],
                    value=1, style={'width': '100%'}),],style={'margin': '10px'}),
            html.Div([
                html.Label("MARRIAGE:", style={'padding': '5px'}),
                dcc.Dropdown(id='input-marriage',
                    options=[
                        {'label': 'Casado', 'value': 1},
                        {'label': 'Soltero', 'value': 2},
                        {'label': 'Otros', 'value': 3}
                    ],
                    value=1, style={'width': '100%'}),],style={'margin': '10px'}),
        ], style={'margin': '20px'}),
        ], style={'flex': '1'}),    
    html.Div([
        html.Div([
            html.Div([
                html.Label("Historial de pago para Noviembre 2023:"),
                dcc.Dropdown(id='input-p1', options=[
                    {'label': 'Pago adelantado', 'value': -1},
                    {'label': 'Pago al día', 'value': 0},
                    {'label': 'Pago retrasado 1 meses', 'value': 1},
                    {'label': 'Pago retrasado 2 meses', 'value': 2},
                    {'label': 'Pago retrasado 3 meses', 'value': 3},
                    {'label': 'Pago retrasado 4 meses', 'value': 4},
                    {'label': 'Pago retrasado 5 meses', 'value': 5},
                    {'label': 'Pago retrasado 6 meses', 'value': 6},
                    {'label': 'Pago retrasado 7 meses', 'value': 7},
                    {'label': 'Pago retrasado 8 meses', 'value': 8},
                    {'label': 'Pago retrasado 9 o más meses', 'value': 9}
                ], value=1, style={'width': '100%'}),],style={'margin': '10px'}),
            html.Div([
                html.Label("Historial de pago para Diciembre 2023:"),
                dcc.Dropdown(id='input-p2', options=[
                {'label': 'Pago adelantado', 'value': -1},
                {'label': 'Pago al día', 'value': 0},
                {'label': 'Pago retrasado 1 meses', 'value': 1},
                {'label': 'Pago retrasado 2 meses', 'value': 2},
                {'label': 'Pago retrasado 3 meses', 'value': 3},
                {'label': 'Pago retrasado 4 meses', 'value': 4},
                {'label': 'Pago retrasado 5 meses', 'value': 5},
                {'label': 'Pago retrasado 6 meses', 'value': 6},
                {'label': 'Pago retrasado 7 meses', 'value': 7},
                {'label': 'Pago retrasado 8 meses', 'value': 8},
                {'label': 'Pago retrasado 9 o más meses', 'value': 9}
                ], value=1, style={'width': '100%'}),],style={'margin': '10px'}),
            html.Div([
                html.Label("Historial de pago para Enero 2024:"),
                dcc.Dropdown(id='input-p3', options=[
                {'label': 'Pago adelantado', 'value': -1},
                {'label': 'Pago al día', 'value': 0},
                {'label': 'Pago retrasado 1 meses', 'value': 1},
                {'label': 'Pago retrasado 2 meses', 'value': 2},
                {'label': 'Pago retrasado 3 meses', 'value': 3},
                {'label': 'Pago retrasado 4 meses', 'value': 4},
                {'label': 'Pago retrasado 5 meses', 'value': 5},
                {'label': 'Pago retrasado 6 meses', 'value': 6},
                {'label': 'Pago retrasado 7 meses', 'value': 7},
                {'label': 'Pago retrasado 8 meses', 'value': 8},
                {'label': 'Pago retrasado 9 o más meses', 'value': 9}
                ], value=1, style={'width': '100%'}),],style={'margin': '10px'}),
            html.Div([
                html.Label("Historial de pago para Febrero 2024:"),
                dcc.Dropdown(id='input-p4', options=[
                {'label': 'Pago adelantado', 'value': -1},
                {'label': 'Pago al día', 'value': 0},
                {'label': 'Pago retrasado 1 meses', 'value': 1},
                {'label': 'Pago retrasado 2 meses', 'value': 2},
                {'label': 'Pago retrasado 3 meses', 'value': 3},
                {'label': 'Pago retrasado 4 meses', 'value': 4},
                {'label': 'Pago retrasado 5 meses', 'value': 5},
                {'label': 'Pago retrasado 6 meses', 'value': 6},
                {'label': 'Pago retrasado 7 meses', 'value': 7},
                {'label': 'Pago retrasado 8 meses', 'value': 8},
                {'label': 'Pago retrasado 9 o más meses', 'value': 9}
                ], value=1, style={'width': '100%'}),],style={'margin': '10px'}),
            html.Div([
                html.Label("Historial de pago para Marzo 2024:"),
                dcc.Dropdown(id='input-p5', options=[
                {'label': 'Pago adelantado', 'value': -1},
                {'label': 'Pago al día', 'value': 0},
                {'label': 'Pago retrasado 1 meses', 'value': 1},
                {'label': 'Pago retrasado 2 meses', 'value': 2},
                {'label': 'Pago retrasado 3 meses', 'value': 3},
                {'label': 'Pago retrasado 4 meses', 'value': 4},
                {'label': 'Pago retrasado 5 meses', 'value': 5},
                {'label': 'Pago retrasado 6 meses', 'value': 6},
                {'label': 'Pago retrasado 7 meses', 'value': 7},
                {'label': 'Pago retrasado 8 meses', 'value': 8},
                {'label': 'Pago retrasado 9 o más meses', 'value': 9}
                ], value=1, style={'width': '100%'}),],style={'margin': '10px'}),
            html.Div([
                html.Label("Historial de pago para Abril 2024:"),
                dcc.Dropdown(id='input-p6', options=[
                {'label': 'Pago adelantado', 'value': -1},
                {'label': 'Pago al día', 'value': 0},
                {'label': 'Pago retrasado 1 meses', 'value': 1},
                {'label': 'Pago retrasado 2 meses', 'value': 2},
                {'label': 'Pago retrasado 3 meses', 'value': 3},
                {'label': 'Pago retrasado 4 meses', 'value': 4},
                {'label': 'Pago retrasado 5 meses', 'value': 5},
                {'label': 'Pago retrasado 6 meses', 'value': 6},
                {'label': 'Pago retrasado 7 meses', 'value': 7},
                {'label': 'Pago retrasado 8 meses', 'value': 8},
                {'label': 'Pago retrasado 9 o más meses', 'value': 9}
                ], value=1)], style={'margin': '10px'}),
            ], style={'flex': '1'}),
        html.Div([
            html.Div([
                html.Label("Monto de deuda para Noviembre 2023:"),
                dcc.Input(id='input-bill1', type='number', value=0)], style={'margin': '10px'}),
            html.Div([
                html.Label("Monto de deuda para Diciembre 2023:"),
                dcc.Input(id='input-bill2', type='number', value=0)], style={'margin': '10px'}),
            html.Div([
                html.Label("Monto de deuda para Enero 2024:"),
                dcc.Input(id='input-bill3', type='number', value=0)], style={'margin': '10px'}),
            html.Div([
                html.Label("Monto de deuda para Febrero 2024:"),
                dcc.Input(id='input-bill4', type='number', value=0)], style={'margin': '10px'}),
            html.Div([
                html.Label("Monto de deuda para Marzo 2024:"),
                dcc.Input(id='input-bill5', type='number', value=0)], style={'margin': '10px'}),
            html.Div([
                html.Label("Monto de deuda para Abril 2024:"),
                dcc.Input(id='input-bill6', type='number', value=0)], style={'margin': '10px'}),
            ], style={'flex': '1'}),
            # Repetir para cada una de las variables restantes
        html.Div([
            # Repetir para cada una de las variables restantes
            html.Div([
                html.Label("Monto del pago para Noviembre 2023:"),
                dcc.Input(id='input-paydone1', type='number', value=0)], style={'margin': '10px'}),
            html.Div([
                html.Label("Monto del pago para Diciembre 2023:"),
                dcc.Input(id='input-paydone2', type='number', value=0)], style={'margin': '10px'}),
            html.Div([
                html.Label("Monto del pago para Enero 2024:"),
                dcc.Input(id='input-paydone3', type='number', value=0)], style={'margin': '10px'}),
            html.Div([
                html.Label("Monto del pago para Febrero 2024:"),
                dcc.Input(id='input-paydone4', type='number', value=0)], style={'margin': '10px'}),
            html.Div([
                html.Label("Monto del pago para Marzo 2024:"),
                dcc.Input(id='input-paydone5', type='number', value=0)], style={'margin': '10px'}),
            html.Div([
                html.Label("Monto del pago para Abril 2024:"),
                dcc.Input(id='input-paydone6', type='number', value=0)], style={'margin': '10px'})
                ], style={'flex': '1'}),
            # Continúa añadiendo inputs para PAY_3, PAY_4, ..., PAY_AMT6
            ], style={'display': 'flex', 'justify-content': 'space-between', 'border': '2px solid #ccc', 'padding': '20px', 'border-radius': '5px', 'box-shadow': '2px 2px 10px #ccc'}),
            html.Button('Predecir', id='predict-button'),
            html.Div(id='prediction-result')
        ])

# Callbacks para el mapa de calor, histograma y predicción
@app.callback(
    Output('correlation-heatmap', 'children'),
    [Input('var-select', 'value')]
)
def update_heatmap(selected_vars):
    filtered_data = data[selected_vars]
    corr_matrix = filtered_data.corr()
    plt.figure(figsize=(10, 10))
    sns.heatmap(corr_matrix, annot=True, cmap='viridis', fmt=".2f")
    buffer = io.BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    image_png = buffer.getvalue()
    graph = html.Img(src='data:image/png;base64,{}'.format(base64.b64encode(image_png).decode()), style={'width': '100%'})
    return graph

@app.callback(
    Output('frequency-histogram', 'children'),
    [Input('histogram-var-select', 'value'),
     Input('age-range-slider', 'value')]
)
def update_histogram(selected_var, age_range):
    # Diccionarios para mapear los valores numéricos a etiquetas
    sex_map = {1: 'Masculino', 2: 'Femenino'}
    education_map = {1: 'Postgrado', 2: 'Universitario', 3: 'Secundaria', 4: 'Otros'}
    marriage_map = {1: 'Casado', 2: 'Soltero', 3: 'Otros'}
    pay_status_map = {-1: 'Pago adelantado', 0: 'Pago puntual', 1: 'Retraso 1 mes', 2: 'Retraso 2 meses', 3: 'Retraso 3 meses', 4: 'Retraso 4 meses', 5: 'Retraso 5 meses', 6: 'Retraso 6 meses', 7: 'Retraso 7 meses', 8: 'Retraso 8 meses', 9: 'Retraso 9 o más meses'}
    
    # Filtrar datos según rango de edad
    filtered_data = data[(data['AGE'] >= age_range[0]) & (data['AGE'] <= age_range[1])]

    # Aplicar mapeo si la variable seleccionada es una de las categorías
    if selected_var == 'SEX':
        filtered_data[selected_var] = filtered_data[selected_var].map(sex_map)
    elif selected_var == 'EDUCATION':
        filtered_data[selected_var] = filtered_data[selected_var].map(education_map)
    elif selected_var == 'MARRIAGE':
        filtered_data[selected_var] = filtered_data[selected_var].map(marriage_map)
    elif selected_var in ['PAY_0', 'PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6']:
        filtered_data[selected_var] = filtered_data[selected_var].map(pay_status_map)
    
    # Crear histograma
    plt.figure(figsize=(10, 6))
    sns.histplot(filtered_data[selected_var], bins=20, color='blue', alpha=0.7)
    plt.title(f'Histograma de {selected_var}')
    plt.xlabel(selected_var)
    plt.ylabel('Frecuencia')
    buffer = io.BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    image_png = buffer.getvalue()
    graph = html.Img(src='data:image/png;base64,{}'.format(base64.b64encode(image_png).decode()), style={'width': '100%'})
    
    return graph

@app.callback(
    Output('value-dropdown', 'options'),
    [Input('category-dropdown', 'value')]
)
def set_values_options(selected_category):
    if selected_category == 'SEX':
        return [{'label': 'Masculino', 'value': 1}, {'label': 'Femenino', 'value': 2}]
    elif selected_category == 'EDUCATION':
        return [{'label': 'Postgrado', 'value': 1}, {'label': 'Universitario', 'value': 2}, {'label': 'Secundaria', 'value': 3}, {'label': 'Otros', 'value': 4}]
    elif selected_category == 'MARRIAGE':
        return [{'label': 'Casado', 'value': 1}, {'label': 'Soltero', 'value': 2}, {'label': 'Otros', 'value': 3}]
    else:
        return []

@app.callback(
    Output('pie-chart-container', 'children'),
    [Input('category-dropdown', 'value'),
     Input('value-dropdown', 'value')]
)
def update_pie_chart(category, value):
    if value is not None:
        filtered_data = data[data[category] == value]
        pie_chart = px.pie(filtered_data, names='default payment next month', title=f'Distribución de Incumplimientos para {category} = {value}')
        return dcc.Graph(figure=pie_chart)
    return html.Div()



@app.callback(
    Output('prediction-result', 'children'),
    [Input('predict-button', 'n_clicks')],
    [dash.dependencies.State('input-limit_bal', 'value'),
     dash.dependencies.State('input-sex', 'value'),
     dash.dependencies.State('input-education', 'value'),
     dash.dependencies.State('input-marriage', 'value'),
     dash.dependencies.State('input-age', 'value'),
     # Añadir los estados para los inputs restantes
     dash.dependencies.State('input-p1', 'value'),
     dash.dependencies.State('input-p2', 'value'),
     dash.dependencies.State('input-p3', 'value'),
     dash.dependencies.State('input-p4', 'value'),
     dash.dependencies.State('input-p5', 'value'),
     dash.dependencies.State('input-p6', 'value'),
     dash.dependencies.State('input-bill1', 'value'),
     dash.dependencies.State('input-bill2', 'value'),
     dash.dependencies.State('input-bill3', 'value'),
     dash.dependencies.State('input-bill4', 'value'),
     dash.dependencies.State('input-bill5', 'value'),
     dash.dependencies.State('input-bill6', 'value'),
     dash.dependencies.State('input-paydone1', 'value'),
     dash.dependencies.State('input-paydone2', 'value'),
     dash.dependencies.State('input-paydone3', 'value'),
     dash.dependencies.State('input-paydone4', 'value'),
     dash.dependencies.State('input-paydone5', 'value'),
     dash.dependencies.State('input-paydone6', 'value'),
     ]
)
def predict(n_clicks, limit_bal, sex, education, marriage, age, pay_0, pay_2, pay_3, pay_4, pay_5, pay_6, bill_amt1, bill_amt2, bill_amt3, bill_amt4, bill_amt5, bill_amt6, pay_amt1, pay_amt2, pay_amt3, pay_amt4, pay_amt5, pay_amt6):
    if n_clicks is not None:
        # Aquí se crea el DataFrame con las entradas del usuario
        input_data = np.array([[limit_bal, sex, education, marriage, age, pay_0, pay_2, pay_3, pay_4, pay_5, pay_6, bill_amt1, bill_amt2, bill_amt3, bill_amt4, bill_amt5, bill_amt6, pay_amt1, pay_amt2, pay_amt3, pay_amt4, pay_amt5, pay_amt6]])
        input_df = pd.DataFrame(input_data, columns=['LIMIT_BAL', 'SEX', 'EDUCATION', 'MARRIAGE', 'AGE', 'PAY_0', 'PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6', 'BILL_AMT1', 'BILL_AMT2', 'BILL_AMT3', 'BILL_AMT4', 'BILL_AMT5', 'BILL_AMT6', 'PAY_AMT1', 'PAY_AMT2', 'PAY_AMT3', 'PAY_AMT4', 'PAY_AMT5', 'PAY_AMT6'])
        # Se añaden más características si el modelo las requiere
        # Realizar predicción
        std_scl = StandardScaler()
        std_scl.fit(input_df)
        nuevos_datos_escalados = std_scl.transform(input_df)
        predicciones = model.predict(nuevos_datos_escalados)
        return f'Predicción: {predicciones[0][0]}'

# Ejecutar el servidor
if __name__ == '__main__':
    app.run_server(debug=True)