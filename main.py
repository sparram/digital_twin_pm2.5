import pandas as pd
import torch
import numpy as np
from src.data_loader import DEFAULT_STATIONS, LAT_REF, LON_REF
import pickle

# 1. Cargar el diccionario de vientos
with open('wind_data.pkl', 'rb') as f:
    wind_data_dict = pickle.load(f)

# 2. Cargar tu CSV de PM2.5 de SIATA
csv_path = 'data/Estaciones_PM2.5_2026_06.csv'
df_pm25 = pd.read_csv(csv_path, parse_dates=['fecha_hora'], index_col='fecha_hora')

# 💡 CORRECCIÓN CRUCIAL: Si el índice de PM2.5 es tz-naive, quitamos el tz a Open-Meteo
for code in wind_data_dict:
    if wind_data_dict[code].index.tz is not None:
        wind_data_dict[code].index = wind_data_dict[code].index.tz_localize(None)

# 3. Preparar listas para acumular los tensores de entrenamiento
x_list, y_list, t_list, c_list, u_list, v_list = [], [], [], [], [], []

t0 = df_pm25.index[0]

for code, (lat, lon) in DEFAULT_STATIONS.items():
    if code in df_pm25.columns and code in wind_data_dict:
        # Coordenadas relativas en kilómetros respecto a la referencia
        x_m = (lon - LON_REF) * 111.0 * np.cos(np.radians(LAT_REF))
        y_m = (lat - LAT_REF) * 111.0
        
        # Alinear PM2.5 y Viento por hora de forma segura
        df_merged = pd.DataFrame({
            'pm25': df_pm25[code],
            'u': wind_data_dict[code]['u'],
            'v': wind_data_dict[code]['v']
        }).dropna()
        
        if len(df_merged) > 0:
            time_hours = (df_merged.index - t0).total_seconds() / 3600.0
            
            for t_val, (idx, row) in zip(time_hours, df_merged.iterrows()):
                x_list.append(x_m)
                y_list.append(y_m)
                t_list.append(t_val)
                c_list.append(np.log(row['pm25'] + 1.0)) # Transformación logarítmica
                u_list.append(row['u'])
                v_list.append(row['v'])

# Convertir a Tensores de PyTorch
x_data = torch.tensor(x_list, dtype=torch.float32).unsqueeze(1)
y_data = torch.tensor(y_list, dtype=torch.float32).unsqueeze(1)
t_data = torch.tensor(t_list, dtype=torch.float32).unsqueeze(1)
C_true = torch.tensor(c_list, dtype=torch.float32).unsqueeze(1)
u_true = torch.tensor(u_list, dtype=torch.float32).unsqueeze(1)
v_true = torch.tensor(v_list, dtype=torch.float32).unsqueeze(1)

print(f"-> ¡Listo! Tensores creados con éxito. Muestras totales: {len(x_data)}")


# Definir los límites geográficos aproximados del Valle de Aburrá en km
# Ancho Este-Oeste (~15 km), Largo Sur-Norte (~45 km), Duración en horas del mes
valle_width_km = 15.0
valle_length_km = 45.0
max_time_hours = float(t_data.max().item())

num_collocation_points = 10000

# Puntos aleatorios uniformes en el dominio espaciotemporal
x_col = torch.rand(num_collocation_points, 1) * valle_width_km
y_col = torch.rand(num_collocation_points, 1) * valle_length_km
t_col = torch.rand(num_collocation_points, 1) * max_time_hours

print(f"-> {num_collocation_points} puntos de colocación física generados.")


from src.trainer import train_pinn

# Ejecutar el entrenamiento (en Google Colab con GPU T4 esto volará)
trained_model = train_pinn(
    x_data, y_data, t_data, C_true, u_true, v_true, 
    x_col, y_col, t_col, 
    epochs=1500, 
    lr=0.001
)