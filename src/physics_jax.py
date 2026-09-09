from functools import partial
import jax
import jax.numpy as jnp
from jax.scipy.signal import convolve2d
import os
import numpy as np

WIND_FILE = "data/wind_vectors.npz"

if os.path.exists(WIND_FILE):
    _WIND_DATA = np.load(WIND_FILE)
    U_SERIES = _WIND_DATA["u"]
    V_SERIES = _WIND_DATA["v"]
else:
    U_SERIES = None
    V_SERIES = None

def get_wind_from_data(step_index: int, override_wind: tuple[float, float] = (0.8, 0.3)) -> tuple[float, float]:
    """Retorna el viento. Si se especifica override_wind, ignora los datos en disco."""
    if override_wind is not None:
        return override_wind
    if U_SERIES is None:
        return 0.8, 0.3
    idx = step_index % len(U_SERIES)
    return float(U_SERIES[idx]), float(V_SERIES[idx])
    
_GAUSSIAN_KERNEL_3X3 = jnp.array([
    [1.0, 2.0, 1.0],
    [2.0, 4.0, 2.0],
    [1.0, 2.0, 1.0]
]) / 16.0

@partial(jax.jit, static_argnames=['Nx', 'Ny'])
def step_physics_single(
    c_flat: jnp.ndarray,
    u_wind: float,
    v_wind: float,
    dx: float,
    dy: float,
    dt: float,
    source_flat: jnp.ndarray = None,
    D_diff: float = 0.15,
    Nx: int = 25,
    Ny: int = 30
) -> jnp.ndarray:
    """Avanza la EDP con condiciones de frontera abiertas y término fuente opcional."""
    C = c_flat.reshape((Ny, Nx))
    C_pad = jnp.pad(C, ((1, 1), (1, 1)), mode='edge')
    
    d2C_dx2 = (C_pad[1:-1, 2:] - 2.0 * C + C_pad[1:-1, :-2]) / (dx ** 2)
    d2C_dy2 = (C_pad[2:, 1:-1] - 2.0 * C + C_pad[:-2, 1:-1]) / (dy ** 2)
    
    dC_dx_back = (C - C_pad[1:-1, :-2]) / dx
    dC_dx_fore = (C_pad[1:-1, 2:] - C) / dx
    dC_dx = jnp.where(u_wind > 0, dC_dx_back, dC_dx_fore)
    
    dC_dy_back = (C - C_pad[:-2, 1:-1]) / dy
    dC_dy_fore = (C_pad[2:, 1:-1] - C) / dy
    dC_dy = jnp.where(v_wind > 0, dC_dy_back, dC_dy_fore)
    
    dC_dt = -(u_wind * dC_dx + v_wind * dC_dy) + D_diff * (d2C_dx2 + d2C_dy2)
    
    if source_flat is not None:
        dC_dt = dC_dt + source_flat.reshape((Ny, Nx))

    C_new = C + dt * dC_dt
    return jnp.clip(C_new.flatten(), 0.0)

_step_ensemble_vmap = jax.vmap(
    step_physics_single,
    in_axes=(1, None, None, None, None, None, None, None, None, None),
    out_axes=1
)

@partial(jax.jit, static_argnames=['Nx', 'Ny'])
def forecast_step(
    key: jax.Array,
    ensemble: jnp.ndarray,
    u_wind: float,
    v_wind: float,
    dx: float,
    dy: float,
    dt: float,
    source_flat: jnp.ndarray = None,
    Q_std: float = 0.5,
    D_diff: float = 0.15,
    Nx: int = 25,
    Ny: int = 30
) -> jnp.ndarray:
    """Paso de pronóstico con inclusión de mapa de emisión."""
    ens_phys = _step_ensemble_vmap(
        ensemble, u_wind, v_wind, dx, dy, dt, source_flat, D_diff, Nx, Ny
    )
    
    n_ens = ensemble.shape[1]
    raw_noise = jax.random.normal(key, shape=(Ny, Nx, n_ens))
    
    def smooth_single_member(n_2d):
        return convolve2d(n_2d, _GAUSSIAN_KERNEL_3X3, mode='same')
    
    spatial_noise = jax.vmap(smooth_single_member, in_axes=2, out_axes=2)(raw_noise)
    spatial_noise = spatial_noise.reshape((Nx * Ny, n_ens)) * Q_std
    
    return jnp.clip(ens_phys + spatial_noise, 0.0)