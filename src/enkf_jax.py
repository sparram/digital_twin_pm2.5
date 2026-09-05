from functools import partial
import jax
import jax.numpy as jnp
from src.physics_jax import forecast_step, get_wind_aburra

@jax.jit
def analysis_step(
    key: jax.Array,
    ensemble: jnp.ndarray,
    y_obs: jnp.ndarray,
    H: jnp.ndarray,
    R: jnp.ndarray
) -> jnp.ndarray:
    """Paso de Análisis del EnKF: Corrección del estado usando observaciones en JAX"""
    n_ensemble = ensemble.shape[1]
    p = y_obs.shape[0]
    
    # 1. Media y matriz de anomalías
    x_mean = jnp.mean(ensemble, axis=1, keepdims=True)
    A_prime = ensemble - x_mean
    HA_prime = H @ A_prime
    
    # 2. Covarianzas P_y y P_xy
    Py = (1.0 / (n_ensemble - 1.0)) * (HA_prime @ HA_prime.T) + R
    P_xy = (1.0 / (n_ensemble - 1.0)) * (A_prime @ HA_prime.T)
    
    # 3. Ganancia de Kalman K = P_xy @ Py^(-1)
    # Se resuelve mediante jnp.linalg.solve(Py, P_xy.T).T para evitar la inversión directa
    K = jnp.linalg.solve(Py, P_xy.T).T
    
    # 4. Perturbación estocástica de observaciones
    obs_noise = jax.random.multivariate_normal(key, jnp.zeros(p), R, shape=(n_ensemble,)).T
    d_perturbed = y_obs.reshape(-1, 1) + obs_noise
    
    # 5. Innovación y actualización del ensamble
    innovation = d_perturbed - (H @ ensemble)
    ensemble_updated = ensemble + K @ innovation
    
    return jnp.clip(ensemble_updated, 0.0)

def run_enkf_assimilation(
    key: jax.Array,
    Y_obs: jnp.ndarray,
    timestamps,
    H: jnp.ndarray,
    dx: float,
    dy: float,
    dt: float = 0.05,
    n_ensemble: int = 40,
    Nx: int = 25,
    Ny: int = 30,
    R_std: float = 2.5,
    Q_std: float = 1.5
):
    """Bucle principal de asimilación espacio-temporal secuencial."""
    total_steps, p = Y_obs.shape
    state_dim = Nx * Ny
    
    key, subkey_init = jax.random.split(key)
    
    # Estado inicial estocástico
    init_val = jnp.nanmean(Y_obs[0])
    ensemble = init_val + jax.random.normal(subkey_init, shape=(state_dim, n_ensemble)) * 4.0
    ensemble = jnp.clip(ensemble, 0.0)
    
    R = jnp.eye(p) * (R_std ** 2)
    campo_reconstruido = []
    
    for t in range(total_steps):
        key, subkey_fore, subkey_anal = jax.random.split(key, 3)
        u_wind, v_wind = get_wind_aburra(timestamps[t].hour)
        
        # Paso 1: Pronóstico (Physics + SPDE)
        ensemble = forecast_step(
            subkey_fore, ensemble, u_wind, v_wind, dx, dy, dt, Q_std=Q_std, Nx=Nx, Ny=Ny
        )
        
        # Paso 2: Análisis (Filtro de Kalman)
        ensemble = analysis_step(
            subkey_anal, ensemble, Y_obs[t], H, R
        )
        
        # Almacenar media del análisis
        x_analysis_mean = jnp.mean(ensemble, axis=1).reshape((Ny, Nx))
        campo_reconstruido.append(x_analysis_mean)
        
    return jnp.array(campo_reconstruido), ensemble