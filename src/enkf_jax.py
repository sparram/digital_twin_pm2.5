from functools import partial
import jax
import jax.numpy as jnp
from src.physics_jax import forecast_step, get_wind_from_data

def gaspari_cohn(r: jnp.ndarray, r_cut: float) -> jnp.ndarray:
    """Función de suavizado de Gaspari-Cohn (soporte compacto)."""
    z = (2.0 * jnp.abs(r)) / r_cut
    
    p1 = 1.0 - (5.0/3.0)*z**2 + (5.0/8.0)*z**3 + (1.0/2.0)*z**4 - (1.0/4.0)*z**5
    p2 = 4.0 - 5.0*z + (5.0/3.0)*z**2 + (5.0/8.0)*z**3 - (1.0/2.0)*z**4 + (1.0/12.0)*z**5 - (2.0/3.0)/(z + 1e-6)
    
    val = jnp.where(z < 1.0, p1, jnp.where(z <= 2.0, p2, 0.0))
    return jnp.maximum(0.0, val)

def compute_localization_matrix(X_grid: jnp.ndarray, Y_grid: jnp.ndarray, active_est_km: dict, r_cut: float = 6.0) -> jnp.ndarray:
    """Construye la matriz C de atenuación espacial (N_celdas, p_estaciones)."""
    X_flat = X_grid.ravel()
    Y_flat = Y_grid.ravel()
    
    est_coords = jnp.array(list(active_est_km.values()))
    x_obs = est_coords[:, 0]
    y_obs = est_coords[:, 1]
    
    dx = X_flat[:, None] - x_obs[None, :]
    dy = Y_flat[:, None] - y_obs[None, :]
    dist = jnp.sqrt(dx**2 + dy**2)
    
    return gaspari_cohn(dist, r_cut=r_cut)

@jax.jit
def analysis_step(
    key: jax.Array,
    ensemble: jnp.ndarray,
    y_obs: jnp.ndarray,
    H: jnp.ndarray,
    R: jnp.ndarray,
    C_mat: jnp.ndarray
) -> jnp.ndarray:
    """Paso de Análisis EnKF con Localización de Covarianza por Producto de Schur."""
    n_ensemble = ensemble.shape[1]
    p = y_obs.shape[0]
    
    # 1. Media y matriz de anomalías
    x_mean = jnp.mean(ensemble, axis=1, keepdims=True)
    A_prime = ensemble - x_mean
    HA_prime = H @ A_prime
    
    # 2. Covarianzas P_y y P_xy (Con localización de Schur)
    Py = (1.0 / (n_ensemble - 1.0)) * (HA_prime @ HA_prime.T) + R
    P_xy = (1.0 / (n_ensemble - 1.0)) * (A_prime @ HA_prime.T)
    P_xy_loc = C_mat * P_xy  # <--- Aplicación de Gaspari-Cohn

    # 3. Ganancia de Kalman localizada K = P_xy_loc @ Py^(-1)
    K = jnp.linalg.solve(Py, P_xy_loc.T).T
    
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
    X: jnp.ndarray,
    Y: jnp.ndarray,
    active_est_km: dict,
    dt: float = 0.05,
    n_ensemble: int = 40,
    Nx: int = 25,
    Ny: int = 30,
    R_std: float = 2.5,
    Q_std: float = 1.5,
    r_cut: float = 6.0
):
    """Bucle principal de asimilación con localización espacial."""
    total_steps, p = Y_obs.shape
    state_dim = Nx * Ny
    
    # Precalculo de la matriz C de localización (Estatico para todas las iteraciones)
    C_mat = compute_localization_matrix(X, Y, active_est_km, r_cut=r_cut)
    
    key, subkey_init = jax.random.split(key)
    init_val = jnp.nanmean(Y_obs[0])
    ensemble = init_val + jax.random.normal(subkey_init, shape=(state_dim, n_ensemble)) * 4.0
    ensemble = jnp.clip(ensemble, 0.0)
    
    R = jnp.eye(p) * (R_std ** 2)
    campo_reconstruido = []
    
    for t in range(total_steps):
        key, subkey_fore, subkey_anal = jax.random.split(key, 3)
        u_t, v_t = get_wind_from_data(t)
        
        # Paso 1: Pronóstico
        ensemble = forecast_step(
            subkey_fore, ensemble, u_t, v_t, dx, dy, dt, Q_std=Q_std, Nx=Nx, Ny=Ny
        )
        
        # Paso 2: Análisis (Pasando C_mat)
        ensemble = analysis_step(
            subkey_anal, ensemble, Y_obs[t], H, R, C_mat
        )
        
        x_analysis_mean = jnp.mean(ensemble, axis=1).reshape((Ny, Nx))
        campo_reconstruido.append(x_analysis_mean)
        
    return jnp.array(campo_reconstruido), ensemble