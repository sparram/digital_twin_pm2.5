from functools import partial
import jax
import jax.numpy as jnp
from jax.scipy.signal import convolve2d
import os
import numpy as np

# Cargar los datos de viento guardados previamente
WIND_FILE = "data/wind_vectors.npz"

if os.path.exists(WIND_FILE):
    _WIND_DATA = np.load(WIND_FILE)
    U_SERIES = _WIND_DATA["u"]
    V_SERIES = _WIND_DATA["v"]
else:
    U_SERIES = None
    V_SERIES = None

def get_wind_from_data(step_index: int) -> tuple[float, float]:
    """Retorna (u_wind, v_wind) reales cargados de Open-Meteo para el paso temporal dado."""
    if U_SERIES is None:
        # Fallback a brisa por defecto si no existe el archivo
        return 0.6, 2.2 
    
    # Mapeo del paso de tiempo al índice disponible
    idx = step_index % len(U_SERIES)
    return float(U_SERIES[idx]), float(V_SERIES[idx])

# Kernel Gaussiano 3x3 para estructurar espacialmente el ruido Q
_GAUSSIAN_KERNEL_3X3 = jnp.array([
    [1.0, 2.0, 1.0],
    [2.0, 4.0, 2.0],
    [1.0, 2.0, 1.0]
]) / 16.0

@partial(jax.jit, static_argnames=['Nx', 'Ny'])
def step_physics_single(
    state_flat: jnp.ndarray,  # Contiene [c_flat, s_flat] concatenados
    u_wind: float,
    v_wind: float,
    dx: float,
    dy: float,
    dt: float,
    D_diff: float = 0.15,
    Nx: int = 25,
    Ny: int = 30
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Avanza la física separando el estado en Concentración (c) y Fuente (S)."""
    state_dim = Nx * Ny
    c_flat = state_flat[:state_dim]
    s_flat = state_flat[state_dim:]
    
    C = c_flat.reshape((Ny, Nx))
    S = s_flat.reshape((Ny, Nx))
    
    # 1. Padding de bordes (Neumann)
    C_pad = jnp.pad(C, ((1, 1), (1, 1)), mode='edge')
    
    # 2. Difusión
    d2C_dx2 = (C_pad[1:-1, 2:] - 2.0 * C + C_pad[1:-1, :-2]) / (dx ** 2)
    d2C_dy2 = (C_pad[2:, 1:-1] - 2.0 * C + C_pad[:-2, 1:-1]) / (dy ** 2)
    
    # 3. Advección Upwind
    dC_dx_back = (C - C_pad[1:-1, :-2]) / dx
    dC_dx_fore = (C_pad[1:-1, 2:] - C) / dx
    dC_dx = jnp.where(u_wind > 0, dC_dx_back, dC_dx_fore)
    
    dC_dy_back = (C - C_pad[:-2, 1:-1]) / dy
    dC_dy_fore = (C_pad[2:, 1:-1] - C) / dy
    dC_dy = jnp.where(v_wind > 0, dC_dy_back, dC_dy_fore)
    
    # EDP temporal para C incluyendo la fuente S (¡Sin ruido directo en C!)
    dC_dt = -(u_wind * dC_dx + v_wind * dC_dy) + D_diff * (d2C_dx2 + d2C_dy2) + S
    C_new = jnp.clip(C + dt * dC_dt, 0.0)
    
    # La fuente S evoluciona con persistencia simple (modelo de caminata aleatoria base)
    #S_new = jnp.maximum(0.0, S) # Se mantiene positiva
    # Factor de decaimiento (alpha controla qué tan rápido se "apagan" las fuentes inivas)
    alpha = 0.95 
    
    # La fuente evoluciona decayendo y sumando el ruido espacial suavizado que ya calculas
    S_new = jnp.maximum(0.0, (1.0 - alpha * dt) * S)
    
    return C_new.flatten(), S_new.flatten()

# Vectorización mediante vmap sobre el eje de miembros (axis=1)
_step_ensemble_vmap = jax.vmap(
    step_physics_single,
    in_axes=(1, None, None, None, None, None, None, None, None),
    out_axes=1
)

@partial(jax.jit, static_argnames=['Nx', 'Ny'])
def forecast_step(
    key: jax.Array,
    ensemble: jnp.ndarray, # Dimensión: (2 * Nx * Ny, n_ens)
    u_wind: float,
    v_wind: float,
    dx: float,
    dy: float,
    dt: float,
    Q_std: float = 0.5,
    D_diff: float = 0.15,
    Nx: int = 25,
    Ny: int = 30
) -> jnp.ndarray:
    """Pronóstico con Estado Aumentado y ruido inyectado en la fuente S."""
    n_ens = ensemble.shape[1]
    state_dim = Nx * Ny
    
    # Separar c y s para cada miembro del ensamble mediante vmap o iteración limpia
    # (Aplicamos step_physics_single a cada columna del ensamble)
    def advance_col(col):
        c_new, s_new = step_physics_single(col, u_wind, v_wind, dx, dy, dt, D_diff, Nx, Ny)
        return jnp.concatenate([c_new, s_new])
        
    ens_phys = jax.vmap(advance_col, in_axes=1, out_axes=1)(ensemble)
    
    # Ruido de proceso aplicado únicamente a la evolución de la fuente S
    key_s, key_c = jax.random.split(key)
    raw_noise = jax.random.normal(key_s, shape=(Ny, Nx, n_ens))
    
    def smooth_single_member(n_2d):
        return convolve2d(n_2d, _GAUSSIAN_KERNEL_3X3, mode='same')
        
    spatial_noise_s = jax.vmap(smooth_single_member, in_axes=2, out_axes=2)(raw_noise)
    spatial_noise_s = spatial_noise_s.reshape((state_dim, n_ens)) * Q_std
    
    # Extraer partes física y fuente del pronóstico físico
    c_part = ens_phys[:state_dim, :]
    s_part = ens_phys[state_dim:, :] + spatial_noise_s
    
    # Asegurar positividad en la fuente y concentración
    return jnp.vstack([jnp.clip(c_part, 0.0), jnp.maximum(0.0, s_part)])