from functools import partial
import jax
import jax.numpy as jnp

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
    """Avanza la EDP de advección-difusión 2D para un solo miembro del ensamble."""
    C = c_flat.reshape((Ny, Nx))
    
    # Diferencias finitas centrales con condiciones de contorno periódicas
    d2C_dx2 = (jnp.roll(C, -1, axis=1) - 2.0 * C + jnp.roll(C, 1, axis=1)) / (dx ** 2)
    d2C_dy2 = (jnp.roll(C, -1, axis=0) - 2.0 * C + jnp.roll(C, 1, axis=0)) / (dy ** 2)
    dC_dx   = (jnp.roll(C, -1, axis=1) - jnp.roll(C, 1, axis=1)) / (2.0 * dx)
    dC_dy   = (jnp.roll(C, -1, axis=0) - jnp.roll(C, 1, axis=0)) / (2.0 * dy)
    
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
    """Paso completo de Pronóstico (Forecast Step): EDP Físico + Ruido Estocástico (SPDE)."""
    # 1. Propagación física en paralelo para los N miembros sin bucles for
    ens_phys = _step_ensemble_vmap(
        ensemble, u_wind, v_wind, dx, dy, dt, D_diff, Nx, Ny
    )
    
    # 2. Perturbación estocástica controlada mediante clave JAX PRNG
    noise = jax.random.normal(key, shape=ens_phys.shape) * Q_std
    
    # 3. Restricción de masa no negativa
    return jnp.clip(ens_phys + noise, 0.0)