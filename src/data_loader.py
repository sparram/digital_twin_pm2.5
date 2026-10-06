import openmeteo_requests
import pandas as pd
import numpy as np
import requests_cache
from retry_requests import retry

# 1. Configuración de Estaciones del Valle de Aburrá
LAT_REF, LON_REF = 6.2442, -75.5812
DEFAULT_STATIONS = {
    'SAB-RAME': (6.145500, -75.621260),
    'MED-VILL': (6.261722, -75.551639),
    'MED-SCRI': (6.280500, -75.636600),
    'MED-ARAN': (6.293460, -75.556427),
    'BEL-FEVE': (6.337550, -75.567800),
    'MED-FISC': (6.268789, -75.573708),
    'CEN-TRAF': (6.252561, -75.56958),
    'EST-HOSP': (6.155531, -75.644169),
    'EST-TABL': (6.118056, -75.631943),
    'MED-BEME': (6.243000, -75.612010),
    'MED-SELE': (6.236361, -75.498472),
    'MED-PROV': (6.229778, -75.586052),
    'MED-ALTA': (6.221894, -75.610603),
    'MED-CES':  (6.207386, -75.551851),
    'MED-TESO': (6.199870, -75.560950),
    'ITA-CJUS': (6.185667, -75.597206),
}

def fetch_wind_data(start_date="2026-06-01", end_date="2026-06-30"):
    """Descarga vientos de Open-Meteo para todas las estaciones configuradas."""
    cache_session = requests_cache.CachedSession('.cache', expire_after=3600)
    retry_session = retry(cache_session, retries=5, backoff_factor=0.2)
    openmeteo = openmeteo_requests.Client(session=retry_session)
    
    url = "https://historical-forecast-api.open-meteo.com/v1/forecast"
    wind_data_dict = {}

    for station_code, (lat, lon) in DEFAULT_STATIONS.items():
        params = {
            "latitude": lat,
            "longitude": lon,
            "start_date": start_date,
            "end_date": end_date,
            "hourly": ["wind_direction_10m", "wind_speed_10m"],
        }
        responses = openmeteo.weather_api(url, params=params)
        response = responses[0]
        
        hourly = response.Hourly()
        ws = hourly.Variables(1).ValuesAsNumpy()
        wd = hourly.Variables(0).ValuesAsNumpy()
        
        time_index = pd.date_range(
            start=pd.to_datetime(hourly.Time(), unit="s", utc=True),
            end=pd.to_datetime(hourly.TimeEnd(), unit="s", utc=True),
            freq=pd.Timedelta(seconds=hourly.Interval()),
            inclusive="left"
        )
        
        df_station = pd.DataFrame({'wind_speed': ws, 'wind_direction': wd}, index=time_index)
        
        # Convertir a componentes u, v
        theta = np.radians(df_station['wind_direction'])
        df_station['u'] = -df_station['wind_speed'] * np.sin(theta)
        df_station['v'] = -df_station['wind_speed'] * np.cos(theta)
        
        wind_data_dict[station_code] = df_station

    print(f"-> Vientos extraídos correctamente para {len(wind_data_dict)} estaciones.")
    return wind_data_dict

# Ejemplo de ejecución del módulo de datos (descomenta cuando tengas el CSV de PM2.5 listo)
wind_data = fetch_wind_data()
import pickle

with open('wind_data.pkl', 'wb') as f:
    pickle.dump(wind_data, f)
print("-> Archivo 'wind_data.pkl' guardado localmente.")