"""
val/synthetic_validation.py
---------------------------
Prueba de Gemelo Sintético (Twin Experiment) Físicamente Consistente.
Genera la verdad integrando la ecuación de Advección-Difusión 2D en JAX,
aplica ruido a los sensores virtuales y evalúa el EnKF en prueba ciega.
"""

import sys
from pathlib import Path

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

from src.grid import setup_grid, build_observation_matrix_H
from src.enkf_jax import run_enkf_assimilation


def compute_kge(y_true: np.ndarray, y_pred: np.ndarray):
    std_true, std_pred = np.std(y_true), np.std(y_pred)
    mean_true, mean_pred = np.mean(y_true), np.mean(y_pred)

    if std_true < 1e-6 or std_pred < 1e-6 or abs(mean_true) < 1e-6:
        return -np.inf, 0.0, 0.0, 0.0

    r = np.corrcoef(y_true, y_pred)[0, 1]
    alpha = std_pred / std_true
    beta = mean_pred / mean_true
    kge = 1.0 - np.sqrt((r - 1.0) ** 2 + (alpha - 1.0) ** 2 + (beta - 1.0) ** 2)
    return kge, r, alpha, beta


def compute_willmott_d(y_true: np.ndarray, y_pred: np.ndarray):
    y_bar = np.mean(y_true)
    num = np.sum((y_true - y_pred) ** 2)
    den = np.sum((np.abs(y_pred - y_bar) + np.abs(y_true - y_bar)) ** 2)
    return 1.0 - (num / den) if den >= 1e-8 else 0.0


def generate_pde_truth(X, Y, dx, dy, dt, T_steps, u=0.8, v=0.3, K=0.15):
    """Genera la verdad física con condiciones de frontera abiertas (idénticas a JAX)."""
    Ny, Nx = X.shape
    C = 10.0 + 15.0 * np.exp(-(((X - 3.0)**2 + (Y - 3.0)**2) / 4.0))
    campo_verdadero = []

    for t in range(T_steps):
        # Relleno de bordes idéntico al modelo de JAX
        C_pad = np.pad(C, ((1, 1), (1, 1)), mode='edge')
        
        # Esquema Upwind con fronteras abiertas
        dC_dx_back = (C - C_pad[1:-1, :-2]) / dx
        dC_dx_fore = (C_pad[1:-1, 2:] - C) / dx
        dC_dx = dC_dx_back if u > 0 else dC_dx_fore
        
        dC_dy_back = (C - C_pad[:-2, 1:-1]) / dy
        dC_dy_fore = (C_pad[2:, 1:-1] - C) / dy
        dC_dy = dC_dy_back if v > 0 else dC_dy_fore
        
        # Laplaciano
        d2C_dx2 = (C_pad[1:-1, 2:] - 2.0 * C + C_pad[1:-1, :-2]) / (dx**2)
        d2C_dy2 = (C_pad[2:, 1:-1] - 2.0 * C + C_pad[:-2, 1:-1]) / (dy**2)
        lap = d2C_dx2 + d2C_dy2
        
        # Fuente diurna centrada
        source = 2.0 * np.sin(2 * np.pi * t / 24.0)**2 * np.exp(-(((X - 5.0)**2 + (Y - 5.0)**2) / 2.0))
        
        C = C + dt * (-u * dC_dx - v * dC_dy + K * lap + source)
        C = np.clip(C, a_min=0.0, a_max=None)
        campo_verdadero.append(C.copy())

    return np.array(campo_verdadero)


def main():
    print("=" * 70)
    print("   TWIN EXPERIMENT FÍSICAMENTE CONSISTENTE (ADVECCIÓN-DIFUSIÓN)   ")
    print("=" * 70)

    Nx, Ny = 25, 30
    T_steps = 120
    dt = 0.05
    x, y, X, Y, dx, dy = setup_grid(Nx=Nx, Ny=Ny)

    synthetic_stations = {
        "SINT_NORTE": (5.0, 8.0),
        "SINT_SUR": (5.0, 2.0),
        "SINT_CENTRO": (5.0, 5.0),  # Estación ciega a predecir
        "SINT_ESTE": (8.0, 5.0),
        "SINT_OESTE": (2.0, 5.0),
        "SINT_NE": (7.5, 7.5),
        "SINT_SO": (2.5, 2.5),
    }

    # 1. Generar la verdad física con la PDE
    campo_verdadero = generate_pde_truth(X, Y, dx, dy, dt, T_steps)

    # 2. Muestrear en los sensores y agregar ruido N(0, 0.5^2)
    np.random.seed(42)
    codes = list(synthetic_stations.keys())
    Y_obs = []

    for code in codes:
        x_s, y_s = synthetic_stations[code]
        dist = np.sqrt((X - x_s)**2 + (Y - y_s)**2)
        idx_min = np.unravel_index(np.argmin(dist), X.shape)
        signal = campo_verdadero[:, idx_min[0], idx_min[1]]
        noise = np.random.normal(0, 0.5, size=T_steps)
        Y_obs.append(signal + noise)

    Y_obs_full = np.column_stack(Y_obs)
    timestamps = pd.date_range("2026-01-01", periods=T_steps, freq="h")

    # 3. Aislar la estación central (Hold-Out)
    holdout_code = "SINT_CENTRO"
    holdout_idx = codes.index(holdout_code)
    y_real_holdout = Y_obs_full[:, holdout_idx]

    train_est_km = {k: v for k, v in synthetic_stations.items() if k != holdout_code}
    Y_obs_train = jnp.delete(Y_obs_full, holdout_idx, axis=1)

    H_train = build_observation_matrix_H(train_est_km, x, y, Nx, Ny)

    # 4. Asimilación EnKF
    key = jax.random.PRNGKey(42)
    campo_reconstruido, _ = run_enkf_assimilation(
        key=key,
        Y_obs=Y_obs_train,
        timestamps=timestamps,
        H=H_train,
        dx=dx, dy=dy,
        X=jnp.array(X), Y=jnp.array(Y),
        active_est_km=train_est_km,
        dt=dt,
        n_ensemble=30,
        Nx=Nx, Ny=Ny,
        R_std=0.5, 
        Q_std=0.3,
        r_cut=4.0,
        inflation_factor=1.05
    )

    # 5. Sintetizar serie en el punto ciego
    holdout_est_km = {holdout_code: synthetic_stations[holdout_code]}
    H_holdout = build_observation_matrix_H(holdout_est_km, x, y, Nx, Ny)
    campo_flat = campo_reconstruido.reshape(T_steps, Nx * Ny)
    y_pred_holdout = np.array(jnp.dot(campo_flat, H_holdout.T)).ravel()

    # 6. Métricas
    residuals = y_real_holdout - y_pred_holdout
    rmse = np.sqrt(np.mean(residuals**2))
    mae = np.mean(np.abs(residuals))
    r2 = np.corrcoef(y_real_holdout, y_pred_holdout)[0, 1]**2
    kge, r_corr, alpha_var, beta_bias = compute_kge(y_real_holdout, y_pred_holdout)
    willmott_d = compute_willmott_d(y_real_holdout, y_pred_holdout)

    print(f"\n================ RESULTADOS TWIN EXPERIMENT ({holdout_code}) ================")
    print(f"  • RMSE:         {rmse:.2f} µg/m³")
    print(f"  • MAE:          {mae:.2f} µg/m³")
    print(f"  • R²:           {r2:.3f}")
    print(f"  • KGE:          {kge:.3f}  (r={r_corr:.2f}, α={alpha_var:.2f}, β={beta_bias:.2f})")
    print(f"  • Willmott (d): {willmott_d:.3f}")
    print("=========================================================================\n")

    plt.figure(figsize=(10, 4.5), dpi=120)
    plt.plot(timestamps, y_real_holdout, 'k-o', label=f'Sensor Sintético Real ({holdout_code})', markersize=3)
    plt.plot(timestamps, y_pred_holdout, 'r--s', label='Gemelo Digital EnKF JAX', markersize=3)
    plt.title(f"Twin Experiment Consistente | KGE: {kge:.2f} | R²: {r2:.2f} | RMSE: {rmse:.2f}", fontweight='bold')
    plt.ylabel('PM2.5 Sintético')
    plt.xlabel('Tiempo')
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend()
    plt.tight_layout()

    save_path = RESULTS_DIR / "synthetic_validation.png"
    plt.savefig(save_path, dpi=300)
    plt.show()


if __name__ == "__main__":
    main()