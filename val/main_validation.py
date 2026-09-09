"""
val/main_validation.py
----------------------
Validación Individual Hold-Out sobre datos reales de PM2.5.
Aisla una estación seleccionada, ejecuta la asimilación EnKF + JAX con las demás
y compara la predicción sintetizada contra el sensor real.
Incorpora Kling-Gupta Efficiency (KGE) y el Índice de Acuerdo de Willmott (d).
"""

import sys
from pathlib import Path

# 1. Configuración de rutas (val y src al mismo nivel)
VAL_DIR = Path(__file__).resolve().parent        # Ruta a /val/
ROOT_DIR = VAL_DIR.parent                        # Raíz del proyecto
RESULTS_DIR = VAL_DIR / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Importar funciones de la arquitectura del proyecto
from src.grid import setup_grid, load_and_clean_data, build_observation_matrix_H
from src.enkf_jax import run_enkf_assimilation


def compute_kge(y_true: np.ndarray, y_pred: np.ndarray):
    """
    Calcula el Kling-Gupta Efficiency (KGE) y sus tres componentes:
    - r: Coeficiente de correlación de Pearson.
    - alpha: Ratio de variabilidad (std_pred / std_true).
    - beta: Ratio de sesgo (mean_pred / mean_true).
    """
    std_true = np.std(y_true)
    std_pred = np.std(y_pred)
    mean_true = np.mean(y_true)
    mean_pred = np.mean(y_pred)

    if std_true < 1e-6 or std_pred < 1e-6 or abs(mean_true) < 1e-6:
        return -np.inf, 0.0, 0.0, 0.0

    r = np.corrcoef(y_true, y_pred)[0, 1]
    alpha = std_pred / std_true
    beta = mean_pred / mean_true

    kge = 1.0 - np.sqrt((r - 1.0) ** 2 + (alpha - 1.0) ** 2 + (beta - 1.0) ** 2)
    return kge, r, alpha, beta


def compute_willmott_d(y_true: np.ndarray, y_pred: np.ndarray):
    """
    Calcula el Índice de Acuerdo de Willmott (d).
    Varía entre 0 (sin acuerdo) y 1 (acuerdo perfecto).
    """
    y_bar = np.mean(y_true)
    num = np.sum((y_true - y_pred) ** 2)
    den = np.sum((np.abs(y_pred - y_bar) + np.abs(y_true - y_bar)) ** 2)
    if den < 1e-8:
        return 0.0
    return 1.0 - (num / den)


def main():
    print("=" * 70)
    print("   VALIDACIÓN HOLD-OUT INDIVIDUAL CON DATOS REALES (SIATA)   ")
    print("=" * 70)

    # 1. Parámetros de la malla y simulación
    Nx, Ny = 25, 30
    max_steps = 24 * 7  # 1 semana (168 horas) de datos reales
    csv_path = ROOT_DIR / 'data' / 'Estaciones_PM2.5_2026_06.csv'

    # 2. Configurar Malla 2D y Cargar Datos Reales
    x, y, X, Y, dx, dy = setup_grid(Nx=Nx, Ny=Ny)
    
    Y_obs_full, timestamps, active_est_km, codes = load_and_clean_data(
        csv_path=str(csv_path), max_steps=max_steps
    )

    # 3. SELECCIÓN DE ESTACIÓN A OCULTAR (HOLD-OUT)
    holdout_code = "MED-ARAN"  # Puedes cambiar por "CEN-TRAF", "MED-SCRI", etc.
    
    if holdout_code not in codes:
        raise ValueError(f"La estación '{holdout_code}' no se encuentra en las estaciones activas del CSV.")

    holdout_idx = codes.index(holdout_code)
    
    print(f"\n[1/3] Aislando estación para prueba ciega: '{holdout_code}'")
    y_real_holdout = np.array(Y_obs_full[:, holdout_idx])

    # 4. Filtrar estaciones visibles para el entrenamiento
    train_est_km = {k: v for k, v in active_est_km.items() if k != holdout_code}
    Y_obs_train = jnp.delete(Y_obs_full, holdout_idx, axis=1)

    # Matriz H de entrenamiento mediante interpolación bilineal
    H_train = build_observation_matrix_H(train_est_km, x, y, Nx, Ny)

    # 5. EJECUTAR ASIMILACIÓN JAX
    print("[2/3] Ejecutando asimilación EnKF en JAX con estaciones visibles...")
    key = jax.random.PRNGKey(42)
    
    campo_reconstruido, _ = run_enkf_assimilation(
        key=key,
        Y_obs=Y_obs_train,
        timestamps=timestamps,
        H=H_train,
        dx=dx, dy=dy,
        X=jnp.array(X), Y=jnp.array(Y),
        active_est_km=train_est_km,
        dt=0.05,
        n_ensemble=40,
        Nx=Nx, Ny=Ny,
        R_std=0.5, Q_std=2.5
    )

    # 6. EXTRAER PREDICCIÓN EN LAS COORDENADAS EXACTAS DE LA ESTACIÓN OCULTA
    holdout_est_km = {holdout_code: active_est_km[holdout_code]}
    H_holdout = build_observation_matrix_H(holdout_est_km, x, y, Nx, Ny)

    T_steps = len(timestamps)
    campo_flat = campo_reconstruido.reshape(T_steps, Nx * Ny)
    y_pred_holdout = np.array(jnp.dot(campo_flat, H_holdout.T)).ravel()

    # 7. CÁLCULO DE MÉTRICAS DE VALIDACIÓN
    print("[3/3] Calculando métricas de rendimiento...")
    residuals = y_real_holdout - y_pred_holdout
    rmse = np.sqrt(np.mean(residuals**2))
    mae = np.mean(np.abs(residuals))
    
    if np.std(y_real_holdout) > 1e-6 and np.std(y_pred_holdout) > 1e-6:
        r2 = np.corrcoef(y_real_holdout, y_pred_holdout)[0, 1]**2
    else:
        r2 = 0.0

    # Nuevas métricas hidro-meteorológicas / geofísicas
    kge, r_corr, alpha_var, beta_bias = compute_kge(y_real_holdout, y_pred_holdout)
    willmott_d = compute_willmott_d(y_real_holdout, y_pred_holdout)

    print(f"\n================ RESULTADOS HOLD-OUT ({holdout_code}) ================")
    print(f"  • RMSE:         {rmse:.2f} µg/m³")
    print(f"  • MAE:          {mae:.2f} µg/m³")
    print(f"  • R²:           {r2:.3f}")
    print(f"  • KGE:          {kge:.3f}  (r={r_corr:.2f}, α={alpha_var:.2f}, β={beta_bias:.2f})")
    print(f"  • Willmott (d): {willmott_d:.3f}")
    print("========================================================\n")

    # 8. GENERAR Y GUARDAR GRÁFICA DE VALIDACIÓN
    fig, ax = plt.subplots(figsize=(10, 4.8), dpi=120)
    ax.plot(timestamps, y_real_holdout, 'o-', color='black', label=f'Sensor Real ({holdout_code})', linewidth=1.5, markersize=3.5)
    ax.plot(timestamps, y_pred_holdout, 's--', color='crimson', label='Gemelo Digital JAX (Sintetizado)', linewidth=1.5, markersize=3.5)

    ax.set_ylabel(r'PM2.5 ($\mu g/m^3$)', fontweight='bold')
    ax.set_xlabel('Fecha y Hora', fontweight='bold')
    ax.set_title(
        f"Validación Hold-Out ({holdout_code}) | RMSE: {rmse:.2f} $\mu g/m^3$ | KGE: {kge:.2f} | $R^2$: {r2:.2f}",
        fontweight='bold'
    )
    ax.grid(True, linestyle=':', alpha=0.6)
    ax.legend(loc='upper right')
    plt.tight_layout()

    # Guardar imagen en la carpeta results/
    save_path = RESULTS_DIR / f"validation_{holdout_code}.png"
    plt.savefig(save_path, dpi=300)
    print(f"[OK] Gráfica guardada en: '{save_path}'")
    plt.show()


if __name__ == "__main__":
    main()