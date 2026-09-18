import matplotlib.animation as animation
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter, laplace
import numpy as np

# --- 1. CONFIGURACIÓN DEL SISTEMA ---
nx, ny = 35, 35
dt = 0.1
D = 1.0  # Coeficiente de difusión
DECAY = 0.02  # Pérdida de calor (evita crecimiento sin límite)
n_ensemble = 40  # Miembros del ensamble

# Ubicaciones de los sensores de temperatura (puntos verdes)
# "Mejor caso": malla densa de sensores cubriendo todo el dominio en vez
# de solo 5 puntos. Con N_PER_SIDE=7 -> 49 sensores repartidos uniformemente.
N_PER_SIDE = 7
_margin = 3
_coords_1d = np.linspace(_margin, nx - 1 - _margin, N_PER_SIDE).round().astype(int)
sensor_coords = [(int(r), int(c)) for r in _coords_1d for c in _coords_1d]
sensor_indices = [r * ny + c for r, c in sensor_coords]
n_obs = len(sensor_coords)
R_var = 0.02  # Varianza del ruido del sensor

# --- Parámetros del EnKF corregido ---
ALPHA_RTPS = 0.7  # relajación al spread previo (0=sin relajación, 1=recupera todo el spread previo)
LOC_RADIUS = 10.0  # radio de localización de covarianza (Gaspari-Cohn), en celdas
                    # (con muchos más sensores, un radio más chico ya alcanza y
                    # mantiene mejor los detalles finos del campo)

np.random.seed(42)

# Reales (Desconocidos para el filtro, el filtro solo los adivina)
true_field = np.zeros((nx, ny))
true_source = np.zeros((nx, ny))

# Ensamble para el Filtro de Kalman (Estado ampliado: Temperaturas y Fuentes)
n_state_single = nx * ny
ensemble_T = np.random.normal(0, 0.1, size=(n_ensemble, nx, ny))
ensemble_S = np.random.normal(0, 0.1, size=(n_ensemble, nx, ny))


# --- 2. MODELO FÍSICO Y DE FUENTE ---
def step_physics(field, source):
  # Difusión de calor influenciada por la fuente actual, con un término
  # de pérdida (DECAY) para que el sistema no crezca indefinidamente.
  lap = laplace(field)
  next_field = field + dt * (D * lap + source - DECAY * field)
  return next_field


def step_source(source):
  # La fuente evoluciona con ruido espacialmente correlacionado.
  raw_noise = np.random.normal(0, 1, size=(nx, ny))
  spatial_noise = gaussian_filter(raw_noise, sigma=3.0)
  return 0.95 * source + 0.2 * spatial_noise


# --- 3. LOCALIZACIÓN DE COVARIANZA (Gaspari-Cohn) ---
def gaspari_cohn(dist, radius):
  r = dist / radius
  out = np.zeros_like(r)
  m1 = r <= 1
  m2 = (r > 1) & (r <= 2)
  out[m1] = (
      -0.25 * r[m1] ** 5
      + 0.5 * r[m1] ** 4
      + 0.625 * r[m1] ** 3
      - 5 / 3 * r[m1] ** 2
      + 1
  )
  out[m2] = (
      (r[m2] ** 5) / 12
      - 0.5 * r[m2] ** 4
      + 0.625 * r[m2] ** 3
      + 5 / 3 * r[m2] ** 2
      - 5 * r[m2]
      + 4
      - (2 / 3) / r[m2]
  )
  return out


# Precalculado una sola vez: peso de localización de cada celda del grid
# respecto a cada sensor (mismo peso para el bloque T y el bloque S).
yy, xx = np.mgrid[0:nx, 0:ny]
loc_weights = np.zeros((n_state_single, n_obs))
for k, (r, c) in enumerate(sensor_coords):
  dist = np.sqrt((xx - c) ** 2 + (yy - r) ** 2).ravel()
  loc_weights[:, k] = gaspari_cohn(dist, LOC_RADIUS)
loc_full = np.vstack([loc_weights, loc_weights])  # (2*n_state_single, n_obs)


# --- 4. ASIMILACIÓN DE DATOS (EnKF con Estado Ampliado + Localización + RTPS) ---
def enkf_step(ens_T, ens_S, true_obs):
  X_f = np.vstack(
      [
          ens_T.reshape(n_ensemble, n_state_single).T,
          ens_S.reshape(n_ensemble, n_state_single).T,
      ]
  )  # (2*N, n_ens)

  Y_f = np.zeros((n_obs, n_ensemble))
  for k, (r, c) in enumerate(sensor_coords):
    Y_f[k, :] = ens_T[:, r, c]

  x_mean = X_f.mean(axis=1, keepdims=True)
  y_mean = Y_f.mean(axis=1, keepdims=True)
  X_p = X_f - x_mean
  Y_p = Y_f - y_mean

  PfHt = (X_p @ Y_p.T) / (n_ensemble - 1)
  PfHt *= loc_full  # localización: corta correlaciones espurias lejos del sensor
  HPfHt = (Y_p @ Y_p.T) / (n_ensemble - 1)
  R = np.eye(n_obs) * R_var

  K = PfHt @ np.linalg.inv(HPfHt + R + np.eye(n_obs) * 1e-4)

  X_a = np.zeros_like(X_f)
  for j in range(n_ensemble):
    obs_noise = np.random.normal(0, np.sqrt(R_var), size=n_obs)
    y_j = true_obs + obs_noise
    X_a[:, j] = X_f[:, j] + K @ (y_j - Y_f[:, j])

  # --- RTPS: relaja el spread posterior hacia el spread previo ---
  # Evita tanto el colapso del ensamble (spread -> 0) como la explosión
  # que causaba la inflación multiplicativa cruda.
  std_f = X_f.std(axis=1, keepdims=True)
  std_a = X_a.std(axis=1, keepdims=True)
  x_a_mean = X_a.mean(axis=1, keepdims=True)
  ratio = np.where(std_a > 1e-8, std_f / std_a, 1.0)
  X_a = x_a_mean + (X_a - x_a_mean) * (ALPHA_RTPS * ratio + (1 - ALPHA_RTPS))

  updated_T = X_a[:n_state_single, :].T.reshape(n_ensemble, nx, ny)
  updated_S = X_a[n_state_single:, :].T.reshape(n_ensemble, nx, ny)

  return updated_T, updated_S


# --- 5. CONFIGURACIÓN VISUAL (4 Paneles) ---
fig, axes = plt.subplots(2, 2, figsize=(10, 9))

im1 = axes[0, 0].imshow(
    true_field, cmap='inferno', origin='lower', vmin=-2, vmax=2
)
axes[0, 0].set_title('1. Temperatura Real (Oculta)')
for r, c in sensor_coords:
  axes[0, 0].plot(c, r, 'go', markersize=3, alpha=0.6)

im2 = axes[0, 1].imshow(
    true_source, cmap='plasma', origin='lower', vmin=-1, vmax=1
)
axes[0, 1].set_title('2. Fuente Real (Móvil y Espacial)')

im3 = axes[1, 0].imshow(
    np.mean(ensemble_T, axis=0), cmap='inferno', origin='lower', vmin=-2, vmax=2
)
axes[1, 0].set_title('3. Temperatura Estimada (EnKF)')
for r, c in sensor_coords:
  axes[1, 0].plot(c, r, 'go', markersize=3, alpha=0.6)

im4 = axes[1, 1].imshow(
    np.mean(ensemble_S, axis=0), cmap='plasma', origin='lower', vmin=-1, vmax=1
)
axes[1, 1].set_title('4. Fuente RECONSTRUIDA por el Filtro')

fig.colorbar(im1, ax=axes[0, 0], fraction=0.046, pad=0.04)
fig.colorbar(im2, ax=axes[0, 1], fraction=0.046, pad=0.04)
fig.colorbar(im3, ax=axes[1, 0], fraction=0.046, pad=0.04)
fig.colorbar(im4, ax=axes[1, 1], fraction=0.046, pad=0.04)


# --- 6. BUCLE DE ANIMACIÓN ---
def animate(i):
  global true_field, true_source, ensemble_T, ensemble_S

  # 1. Evoluciona la Realidad
  true_source = step_source(true_source)
  cx = int(nx / 2 + 8 * np.sin(i * 0.1))
  cy = int(ny / 2 + 8 * np.cos(i * 0.1))
  true_source[
      max(0, cx - 3) : min(nx, cx + 3), max(0, cy - 3) : min(ny, cy + 3)
  ] += 1.5

  true_field = step_physics(true_field, true_source)

  true_obs = np.array(
      [true_field[r, c] for r, c in sensor_coords]
  ) + np.random.normal(0, np.sqrt(R_var), size=n_obs)

  # 2. Pronóstico del Ensamble
  for j in range(n_ensemble):
    ensemble_S[j] = step_source(ensemble_S[j])
    ensemble_T[j] = step_physics(ensemble_T[j], ensemble_S[j])

  # 3. Asimilación de Datos
  ensemble_T, ensemble_S = enkf_step(ensemble_T, ensemble_S, true_obs)

  # Actualizar gráficos
  im1.set_array(true_field)
  im1.set_clim(vmin=true_field.min(), vmax=true_field.max())

  im2.set_array(true_source)
  im2.set_clim(vmin=true_source.min(), vmax=true_source.max())

  est_T = np.mean(ensemble_T, axis=0)
  im3.set_array(est_T)
  im3.set_clim(vmin=est_T.min(), vmax=est_T.max())

  est_S = np.mean(ensemble_S, axis=0)
  im4.set_array(est_S)
  im4.set_clim(vmin=est_S.min(), vmax=est_S.max())

  return [im1, im2, im3, im4]


ani = animation.FuncAnimation(
    fig, animate, frames=200, interval=50, blit=False
)
plt.tight_layout()
plt.show()