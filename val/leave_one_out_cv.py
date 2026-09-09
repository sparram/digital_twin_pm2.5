"""
val/leave_one_out_cv.py
-----------------------
Validación Cruzada Leave-One-Out (LOOCV) sobre todas las estaciones de PM2.5.
Evalúa el desempeño del Gemelo Digital estación por estación y genera
una tabla comparativa global con métricas KGE, Willmott d, RMSE, MAE y R².
"""

import sys
from pathlib import Path

# 1. Configuración de rutas (val y src al mismo nivel)
VAL_DIR = Path(__file__).resolve().parent
ROOT_DIR = VAL_DIR.parent
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
    """Calcula Kling-Gupta Efficiency (KGE) y sus componentes (r, alpha, beta)."""
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
    """Calcula el Índice de Acuerdo de Willmott (d)."""
    y_bar = np.mean(y_true)
    num = np.sum((y_true - y_pred) ** 2)
    den = np.sum((np.abs(y_pred - y_bar) + np.abs(y_true - y_bar)) ** 2)
    if den < 1e-8:
        return 0.0
    return 1.0 - (num / den)


def main():
    print("=" * 75)
    print("   VALIDACIÓN CRUZADA LEAVE-ONE-OUT (LOOCV) - RED SIATA PM2.5   ")
    print("=" * 75)

    # 1. Parámetros de la malla y simulación
    Nx, Ny = 25, 30
    max_steps = 24 * 7  # 1 semana (168 horas)
    csv_path = ROOT_DIR / 'data' / 'Estaciones_PM2.5_2026_06.csv'

    # 2. Cargar malla y datos completos
    x, y, X, Y, dx, dy = setup_grid(Nx=Nx, Ny=Ny)
    Y_obs_full, timestamps, active_est_km, codes = load_and_clean_data(
        csv_path=str(csv_path), max_steps=max_steps
    )

    results = []
    T_steps = len(timestamps)

    print(f"\n[+] Iniciando LOOCV sobre {len(codes)} estaciones activas...\n")

    # 3. Bucle de Validación Cruzada
    for idx, holdout_code in enumerate(codes, 1):
        print(f"[{idx}/{len(codes)}] Evaluando estación a ciegas: '{holdout_code}'...")

        holdout_idx = codes.index(holdout_code)
        y_real = np.array(Y_obs_full[:, holdout_idx])

        # Filtrar datos de entrenamiento (excluyendo la estación actual)
        train_est_km = {k: v for k, v in active_est_km.items() if k != holdout_code}
        Y_obs_train = jnp.delete(Y_obs_full, holdout_idx, axis=1)

        H_train = build_observation_matrix_H(train_est_km, x, y, Nx, Ny)

        # Ejecutar asimilación EnKF + JAX
        key = jax.random.PRNGKey(42 + idx)
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
            R_std=1.5, Q_std=3.5
        )

        # Sintetizar serie temporal en la posición exacta de la estación oculta
        holdout_est_km = {holdout_code: active_est_km[holdout_code]}
        H_holdout = build_observation_matrix_H(holdout_est_km, x, y, Nx, Ny)
        campo_flat = campo_reconstruido.reshape(T_steps, Nx * Ny)
        y_pred = np.array(jnp.dot(campo_flat, H_holdout.T)).ravel()

        # Métricas
        residuals = y_real - y_pred
        rmse = np.sqrt(np.mean(residuals ** 2))
        mae = np.mean(np.abs(residuals))
        
        if np.std(y_real) > 1e-6 and np.std(y_pred) > 1e-6:
            r2 = np.corrcoef(y_real, y_pred)[0, 1] ** 2
        else:
            r2 = 0.0

        kge, r_corr, alpha_var, beta_bias = compute_kge(y_real, y_pred)
        willmott_d = compute_willmott_d(y_real, y_pred)

        results.append({
            "Estación": holdout_code,
            "RMSE": rmse,
            "MAE": mae,
            "R2": r2,
            "KGE": kge,
            "r": r_corr,
            "alpha": alpha_var,
            "beta": beta_bias,
            "Willmott_d": willmott_d
        })

    # 4. Consolidar y Guardar Resultados
    df_res = pd.DataFrame(results)
    csv_save_path = RESULTS_DIR / "loocv_results.csv"
    df_res.to_csv(csv_save_path, index=False)

    print("\n" + "=" * 80)
    print("                      RESUMEN GLOBAL DE LOOCV                      ")
    print("=" * 80)
    print(df_res.to_string(index=False))
    print("-" * 80)
    print(f"PROMEDIO RED -> RMSE: {df_res['RMSE'].mean():.2f} µg/m³ | KGE: {df_res['KGE'].mean():.2f} | R²: {df_res['R2'].mean():.2f} | Willmott: {df_res['Willmott_d'].mean():.2f}")
    print("=" * 80)

    # 5. Generar Gráfica Global Comparativa
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True, dpi=120)

    # Gráfico KGE
    colors = ['teal' if k >= 0 else 'crimson' for k in df_res["KGE"]]
    ax1.bar(df_res["Estación"], df_res["KGE"], color=colors, alpha=0.85)
    ax1.axhline(0, color='black', linestyle='--', linewidth=0.8)
    ax1.set_ylabel("KGE (Kling-Gupta)", fontweight='bold')
    ax1.set_title("Validación Cruzada LOOCV - Desempeño Espacial por Estación", fontweight='bold')
    ax1.grid(True, linestyle=':', alpha=0.6)

    # Gráfico RMSE
    ax2.bar(df_res["Estación"], df_res["RMSE"], color='steelblue', alpha=0.85)
    ax2.set_ylabel(r"RMSE ($\mu g/m^3$)", fontweight='bold')
    ax2.set_xlabel("Estaciones SIATA", fontweight='bold')
    ax2.grid(True, linestyle=':', alpha=0.6)
    plt.xticks(rotation=45, ha='right')

    plt.tight_layout()
    fig_save_path = RESULTS_DIR / "loocv_metrics.png"
    plt.savefig(fig_save_path, dpi=300)
    
    print(f"\n[OK] Resultados CSV guardados en: '{csv_save_path}'")
    print(f"[OK] Gráfica comparativa guardada en: '{fig_save_path}'")
    plt.show()


if __name__ == "__main__":
    main()