from functools import partial
import jax
import jax.numpy as jnp
from jax.scipy.signal import convolve2d

# Kernel Gaussiano 3x3 para estructurar espacialmente el ruido Q
_GAUSSIAN_KERNEL_3X3 = jnp.array([
    [1.0, 2.0, 1.0],
    [2.0, 4.0, 2.0],
    [1.0, 2.0, 1.0]
]) / 16.0

def get_wind_aburra(hour: int) -> tuple[float, float]:
    """Retorna las componentes (u_wind, v_wind) en km/h según el ciclo diurno del Aburrá."""
    if 8 <= hour <= 18:
        return 0.6, 2.2  # Brisa diurna Sur -> Norte
    return 0.2, 0.8      # Drenaje nocturno suave

@partial(jax.jit, static_argnames=['Nx', 'Ny'])
def step_physics_single(
    c_flat: jnp.ndarray,
    u_wind: float,
    v_wind: float,
    dx: float,
    dy: float,
    dt: float,
    D_diff: float = 0.15,
    Nx: int = 25,
    Ny: int = 30
) -> jnp.ndarray:
    """Avanza la EDP con condiciones de frontera abiertas (Neumann/Outflow)."""
    C = c_flat.reshape((Ny, Nx))
    
    # 1. Padding 'edge': duplica los bordes (gradiente espacial nulo en frontera, dC/dn = 0)
    C_pad = jnp.pad(C, ((1, 1), (1, 1)), mode='edge')
    
    # 2. Difusión de segundo orden con bordes abiertos
    d2C_dx2 = (C_pad[1:-1, 2:] - 2.0 * C + C_pad[1:-1, :-2]) / (dx ** 2)
    d2C_dy2 = (C_pad[2:, 1:-1] - 2.0 * C + C_pad[:-2, 1:-1]) / (dy ** 2)
    
    # 3. Advección Upwind (el flujo depende de la dirección del viento)
    # Eje X:
    dC_dx_back = (C - C_pad[1:-1, :-2]) / dx
    dC_dx_fore = (C_pad[1:-1, 2:] - C) / dx
    dC_dx = jnp.where(u_wind > 0, dC_dx_back, dC_dx_fore)
    
    # Eje Y:
    dC_dy_back = (C - C_pad[:-2, 1:-1]) / dy
    dC_dy_fore = (C_pad[2:, 1:-1] - C) / dy
    dC_dy = jnp.where(v_wind > 0, dC_dy_back, dC_dy_fore)
    
    # EDP temporal
    dC_dt = -(u_wind * dC_dx + v_wind * dC_dy) + D_diff * (d2C_dx2 + d2C_dy2)
    C_new = C + dt * dC_dt
    
    return jnp.clip(C_new.flatten(), 0.0)

# Vectorización mediante vmap sobre el eje de miembros (axis=1)
_step_ensemble_vmap = jax.vmap(
    step_physics_single,
    in_axes=(1, None, None, None, None, None, None, None, None),
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
    Q_std: float = 1.5,
    D_diff: float = 0.15,
    Nx: int = 25,
    Ny: int = 30
) -> jnp.ndarray:
    """Paso completo de Pronóstico (Forecast Step) con fronteras abiertas y ruido estructurado."""
    # Propagación física
    ens_phys = _step_ensemble_vmap(
        ensemble, u_wind, v_wind, dx, dy, dt, D_diff, Nx, Ny
    )
    
    # Ruido de proceso 2D correlacionado espacialmente
    n_ens = ensemble.shape[1]
    raw_noise = jax.random.normal(key, shape=(Ny, Nx, n_ens))
    
    def smooth_single_member(n_2d):
        return convolve2d(n_2d, _GAUSSIAN_KERNEL_3X3, mode='same')
    
    spatial_noise = jax.vmap(smooth_single_member, in_axes=2, out_axes=2)(raw_noise)
    spatial_noise = spatial_noise.reshape((Nx * Ny, n_ens)) * Q_std
    
    return jnp.clip(ens_phys + spatial_noise, 0.0)