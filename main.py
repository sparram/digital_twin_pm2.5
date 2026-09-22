import time
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

# Módulos originales de tu proyecto
from src.grid import setup_grid, load_and_clean_data, build_observation_matrix_H
from src.enkf_jax import run_enkf_assimilation

# Nuevos módulos de métricas y visualización
from src.metrics import compute_assimilation_metrics, plot_station_validation
from src.visualization import generate_assimilation_gif

def main():
    # 1. Configurar semilla determinista para JAX
    key = jax.random.PRNGKey(42)

    # 2. Configurar Malla 2D y Cargar Datos
    Nx, Ny = 25, 30
    x, y, X, Y, dx, dy = setup_grid(Nx=Nx, Ny=Ny)
    
    csv_path = 'data/Estaciones_PM2.5_2026_06.csv'
    Y_obs, timestamps, active_est_km, codes = load_and_clean_data(
        csv_path=csv_path, max_steps=int(24*7)
    )
    
    # 3. Construir Operador de Observación H
    H = build_observation_matrix_H(active_est_km, x, y, Nx, Ny)
    
    print(f"Malla: {Nx}x{Ny} | Estaciones activas: {len(codes)} | Pasos de tiempo: {len(timestamps)}")
    
    # 4. Paso de calentamiento (JIT Compilation Warmup)
    print("Compilando grafos de cómputo JAX (Warmup)...")
    key, subkey = jax.random.split(key)
    _ = run_enkf_assimilation(
        subkey, Y_obs[:2], timestamps[:2], H, dx, dy, X, Y, active_est_km, n_ensemble=10, Nx=Nx, Ny=Ny
    )
    
    # 5. Ejecución del filtro EnKF completo con benchmarking
    print("Ejecutando asimilación EnKF vectorizada en JAX...")
    start_time = time.time()
    
    key, subkey = jax.random.split(key)
    
    # MODIFICADO: Recibimos los historiales que ahora exporta enkf_jax.py
    hist_concentracion, hist_fuentes, ensemble_final = run_enkf_assimilation(
        subkey, Y_obs, timestamps, H, dx, dy, X, Y, active_est_km,
        dt=0.05, n_ensemble=40, Nx=Nx, Ny=Ny, R_std=1.5, Q_std=0.5
    )
    
    # Nota: Si el campo reconstruido principal se usa para las métricas, 
    # puedes usar directamente hist_concentracion.
    campo_reconstruido = hist_concentracion 

    # Sincronizar llamadas asíncronas de JAX para medir tiempo real de ejecución
    campo_reconstruido.block_until_ready()
    elapsed_time = time.time() - start_time
    
    print(f"\n¡Asimilación completada con éxito!")
    print(f"Tiempo total de ejecución JAX: {elapsed_time:.4f} segundos")
    print(f"Velocidad promedio: {len(timestamps) / elapsed_time:.2f} pasos/segundo")

    # 6. Calcular Métricas de Rendimiento (RMSE y R2)
    Y_pred, rmse_per_step, rmse_global, r2_global = compute_assimilation_metrics(
        Y_obs, campo_reconstruido, H
    )
    
    print("\n" + "="*45)
    print("       MÉTRICAS DE ASIMILACIÓN (EnKF + JAX)    ")
    print("="*45)
    print(f"  RMSE Global   : {rmse_global:.3f} µg/m³")
    print(f"  R² Global     : {r2_global:.4f} ({r2_global*100:.2f}% de varianza explicada)")
    print("="*45 + "\n")

    # 7. Graficar comparación temporal en la estación ITA-CJUS (Itagüí)
    plot_station_validation(timestamps, np.array(Y_obs), Y_pred, codes, target_code='MED-ARAN')
    plot_station_validation(timestamps, np.array(Y_obs), Y_pred, codes, target_code='SAB-RAME')
    plot_station_validation(timestamps, np.array(Y_obs), Y_pred, codes, target_code='CEN-TRAF')

    # 8. Exportar GIF Animado del Gemelo Digital (Dual: Concentración vs Fuente)
    generate_assimilation_gif(
        campo_reconstruido=hist_concentracion, 
        campo_fuentes=hist_fuentes, 
        X=X, Y=Y, 
        active_est_km=active_est_km, 
        timestamps=timestamps, 
        Y_obs=Y_obs, # Usamos el Y_obs original que cargaste en el paso 2
        output_gif='aburra_dual_enkf.gif'
    )

if __name__ == '__main__':
    main()