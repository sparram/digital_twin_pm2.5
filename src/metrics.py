import numpy as np
import matplotlib.pyplot as plt
import jax.numpy as jnp

def compute_assimilation_metrics(Y_obs, campo_reconstruido, H):
    total_steps, p = Y_obs.shape
    y_obs_np = np.array(Y_obs)
    campo_flat = np.array(campo_reconstruido).reshape((total_steps, -1))
    
    Y_pred = (np.array(H) @ campo_flat.T).T
    
    rmse_per_step = np.sqrt(np.mean((y_obs_np - Y_pred) ** 2, axis=1))
    rmse_global = float(np.sqrt(np.mean((y_obs_np - Y_pred) ** 2)))
    
    ss_res = np.sum((y_obs_np - Y_pred) ** 2)
    ss_tot = np.sum((y_obs_np - np.mean(y_obs_np)) ** 2)
    r2_global = float(1.0 - (ss_res / ss_tot))
    
    return Y_pred, rmse_per_step, rmse_global, r2_global

def plot_station_validation(timestamps, Y_obs, Y_pred, codes, target_code='ITA-CJUS'):
    if target_code not in codes:
        target_code = codes[0]
        
    idx = codes.index(target_code)
    
    plt.figure(figsize=(10, 4.5), dpi=120)
    plt.plot(timestamps, Y_obs[:, idx], 'k.-', label=f'Observación Real SIATA ({target_code})', alpha=0.75)
    plt.plot(timestamps, Y_pred[:, idx], 'r--', label=f'Asimilado EnKF (JAX)', linewidth=2)
    
    plt.title(f"Validación Temporal de Asimilación EnKF - Estación {target_code}", fontsize=11, fontweight='bold')
    plt.ylabel(r'Concentración $PM_{2.5}$ ($\mu g / m^3$)', fontsize=10)
    plt.xlabel('Fecha / Hora', fontsize=10)
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend(frameon=True, facecolor='white', edgecolor='none')
    plt.tight_layout()
    plt.show()