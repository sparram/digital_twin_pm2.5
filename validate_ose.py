import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt

from src.grid import setup_grid, load_and_clean_data, build_observation_matrix_H
from src.enkf_jax import run_enkf_assimilation

def main():
    # 1. Configuración inicial
    key = jax.random.PRNGKey(42)
    Nx, Ny = 25, 30
    x, y, X, Y, dx, dy = setup_grid(Nx=Nx, Ny=Ny)
    
    csv_path = 'data/Estaciones_PM2.5_2026_06.csv'
    Y_obs_full, timestamps, active_est_km_full, codes_full = load_and_clean_data(
        csv_path=csv_path, max_steps=int(24*7)
    )
    
    # 2. Definir cuáles estaciones vamos a OCULTAR (Validación OSE)
    # Escoge 2 o 3 estaciones representativas
    hidden_codes = ['ITA-CJUS', 'MED-ARAN', 'SAB-RAME']
    
    print(f"Estaciones ocultas para validación OSE: {hidden_codes}")
    
    # Separar datos y estaciones activas vs ocultas
    active_est_km = {}
    hidden_est_km = {}
    hidden_indices = []
    
    for code, coords in active_est_km_full.items():
        if code in hidden_codes:
            hidden_est_km[code] = coords
            # Encontrar su índice original en la lista codes_full
            hidden_indices.append(codes_full.index(code))
        else:
            active_est_km[code] = coords
            
    codes_active = [c for c in codes_full if c not in hidden_codes]
    
    # Extraer las observaciones reales de las estaciones ocultas (para comparar después)
    Y_obs_full_np = np.array(Y_obs_full)
    Y_obs_hidden = Y_obs_full_np[:, hidden_indices] # Forma: (T, n_hidden)
    
    # Filtrar Y_obs solo para las estaciones que SÍ van al filtro
    active_indices = [codes_full.index(c) for c in codes_active]
    Y_obs_assim = Y_obs_full_np[:, active_indices]
    Y_obs_assim_jnp = jnp.array(Y_obs_assim)

    # 3. Construir el operador H exclusivo para las estaciones visibles
    H_active = build_observation_matrix_H(active_est_km, x, y, Nx, Ny)
    
    print(f"Malla: {Nx}x{Ny} | Estaciones para asimilar: {len(codes_active)} | Estaciones ocultas: {len(hidden_codes)}")

    # 4. Ejecutar EnKF usando ÚNICAMENTE las estaciones visibles
    print("Ejecutando asimilación EnKF (sin las estaciones ocultas)...")
    key, subkey = jax.random.split(key)
    
    hist_concentracion, hist_fuentes, _ = run_enkf_assimilation(
        subkey, Y_obs_assim_jnp, timestamps, H_active, dx, dy, X, Y, active_est_km,
        dt=0.05, n_ensemble=40, Nx=Nx, Ny=Ny, R_std=1.5, Q_std=3.5, r_cut=10.0
    )
    
    # 5. Evaluar el modelo en las estaciones ocultas
    print("\nEvaluando capacidad predictiva en estaciones ocultas...")
    
    # Para cada estación oculta, extraemos su coordenada (x_h, y_h) 
    # y buscamos la celda de la malla más cercana en hist_concentracion (T, Ny, Nx)
    fig, axes = plt.subplots(len(hidden_codes), 1, figsize=(12, 4 * len(hidden_codes)), sharex=True)
    if len(hidden_codes) == 1:
        axes = [axes]

    for idx, code in enumerate(hidden_codes):
        hx, hy = hidden_est_km[code]
        # Encontrar índices de celda más cercanos
        ix = np.argmin(np.abs(x - hx))
        iy = np.argmin(np.abs(y - hy))
        
        # Pronóstico del modelo en esa ubicación exacta a lo largo del tiempo
        # hist_concentracion tiene forma (T, Ny, Nx), ojo con el orden de los índices (iy, ix)
        pred_hidden = hist_concentracion[:, iy, ix]
        real_hidden = Y_obs_hidden[:, idx]
        
        # Calcular RMSE individual para esta estación oculta
        valid_mask = ~np.isnan(real_hidden)
        rmse_station = np.sqrt(np.mean((pred_hidden[valid_mask] - real_hidden[valid_mask])**2))
        
        print(f" -> Estación Oculta {code}: RMSE OSE = {rmse_station:.3f} µg/m³")
        
        # Graficar comparación
        axes[idx].plot(timestamps, real_hidden, 'r.-', label=f'Real (Oculto): {code}', alpha=0.7)
        axes[idx].plot(timestamps, pred_hidden, 'b-', label=f'Predicción Modelo (Sin verla)', alpha=0.9)
        axes[idx].set_ylabel('PM2.5 ($\mu$g/m³)')
        axes[idx].set_title(f'Validación OSE en Estación: {code} (RMSE = {rmse_station:.3f})')
        axes[idx].legend(loc='upper right')
        axes[idx].grid(True, linestyle='--', alpha=0.5)

    plt.xlabel('Tiempo')
    plt.tight_layout()
    plt.savefig('ose_validation_results.png', dpi=300)
    print("\n¡Gráfica de validación OSE guardada como 'ose_validation_results.png'!")
    plt.show()

if __name__ == '__main__':
    main()