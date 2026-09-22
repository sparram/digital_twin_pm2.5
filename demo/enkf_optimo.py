"""
EnKF con la red de sensores ÓPTIMA (cargada desde sensor_placement_greedy.py)
================================================================================
Carga las coordenadas guardadas en sensores_optimos.npz (generado por
sensor_placement_greedy.py) y corre el EnKF real con exactamente esa red de
sensores, mostrando en vivo el spread empírico vs el error a priori esperado
(igual que osse_vs_enkf_animado.py, pero con la red optimizada en vez de una
malla uniforme).

Requiere haber corrido antes sensor_placement_greedy.py, que deja el archivo:
    /mnt/user-data/outputs/sensores_optimos.npz
"""

import time
import numpy as np
import matplotlib.animation as animation
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter, laplace

# ================== CARGAR LA RED DE SENSORES ÓPTIMA ==================
data = np.load('sensores_optimos.npz')
sensor_coords = [tuple(int(v) for v in row) for row in data['coords']]
nx, ny = int(data['nx']), int(data['ny'])
n_obs = len(sensor_coords)
print(f"Cargados {n_obs} sensores óptimos, grid {nx}x{ny}.")

# ================== CONFIGURACIÓN (debe coincidir con la usada al calcular
#                     la red óptima, salvo N_ensemble/ALPHA_RTPS/LOC_RADIUS
#                     que son parámetros del EnKF, no de la física) ==================
dt, D, DECAY = 0.1, 1.0, 0.02
n_ensemble = 40
n_state_single = nx * ny
R_var = 0.02

ALPHA_RTPS = 0.7
LOC_RADIUS = 8.0   # con más sensores y más cerca entre sí, un radio algo menor
                    # que en la malla dispersa suele andar bien; ajustable.

np.random.seed(42)


def idx(r, c):
  return r * ny + c


# ============================================================================
# PARTE 1 -- OSSE a priori (Riccati) PARA ESTA red de sensores específica
# ============================================================================
def calcular_osse_a_priori():
  print("Calculando OSSE a priori (Riccati) para la red óptima...")
  t0 = time.time()
  n = n_state_single

  L = np.zeros((n, n))
  for r in range(nx):
    for c in range(ny):
      i = idx(r, c)
      vecinos = []
      for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        rr, cc = r + dr, c + dc
        if 0 <= rr < nx and 0 <= cc < ny:
          vecinos.append(idx(rr, cc))
      L[i, i] = -len(vecinos)
      for j in vecinos:
        L[i, j] = 1.0

  M_T = np.eye(n) + dt * (D * L - DECAY * np.eye(n))
  M_TS = dt * np.eye(n)
  M_S = 0.95 * np.eye(n)
  M = np.block([[M_T, M_TS], [np.zeros((n, n)), M_S]])

  n_mc = 1500
  samples = np.zeros((n_mc, n))
  for k in range(n_mc):
    raw = np.random.normal(0, 1, size=(nx, ny))
    samples[k] = (0.2 * gaussian_filter(raw, sigma=3.0)).ravel()
  Q_S = np.cov(samples.T)
  Q = np.zeros((2 * n, 2 * n))
  Q[n:, n:] = Q_S

  H = np.zeros((n_obs, 2 * n))
  for k, (r, c) in enumerate(sensor_coords):
    H[k, idx(r, c)] = 1.0
  R = np.eye(n_obs) * R_var

  P = np.eye(2 * n) * 1.0
  for it in range(120):
    P_f = M @ P @ M.T + Q
    S = H @ P_f @ H.T + R
    K = P_f @ H.T @ np.linalg.inv(S)
    P = P_f - K @ H @ P_f

  sigma_T_priori = np.sqrt(np.diag(P)[:n]).reshape(nx, ny)
  sigma_S_priori = np.sqrt(np.diag(P)[n:]).reshape(nx, ny)
  DFS = np.trace(K @ H)

  print(f"  Listo en {time.time()-t0:.1f}s.")
  print(f"  n_obs = {n_obs}, DFS = {DFS:.2f}")
  print(f"  Error a priori esperado (fuente):      media={sigma_S_priori.mean():.4f}, "
        f"max={sigma_S_priori.max():.4f}")
  print(f"  Error a priori esperado (temperatura): media={sigma_T_priori.mean():.4f}")
  return sigma_T_priori, sigma_S_priori


sigma_T_priori, sigma_S_priori = calcular_osse_a_priori()


# ============================================================================
# PARTE 2 -- EnKF real con la red óptima
# ============================================================================
def gaspari_cohn(dist, radius):
  r = dist / radius
  out = np.zeros_like(r)
  m1 = r <= 1
  m2 = (r > 1) & (r <= 2)
  out[m1] = (-0.25*r[m1]**5 + 0.5*r[m1]**4 + 0.625*r[m1]**3 - 5/3*r[m1]**2 + 1)
  out[m2] = ((r[m2]**5)/12 - 0.5*r[m2]**4 + 0.625*r[m2]**3
             + 5/3*r[m2]**2 - 5*r[m2] + 4 - (2/3)/r[m2])
  return out


yy, xx = np.mgrid[0:nx, 0:ny]
loc_weights = np.zeros((n_state_single, n_obs))
for k, (r, c) in enumerate(sensor_coords):
  dist = np.sqrt((xx - c) ** 2 + (yy - r) ** 2).ravel()
  loc_weights[:, k] = gaspari_cohn(dist, LOC_RADIUS)
loc_full = np.vstack([loc_weights, loc_weights])


def step_physics(field, source):
  lap = laplace(field)
  return field + dt * (D * lap + source - DECAY * field)


def step_source(source):
  raw_noise = np.random.normal(0, 1, size=(nx, ny))
  return 0.95 * source + 0.2 * gaussian_filter(raw_noise, sigma=3.0)


def enkf_step(ens_T, ens_S, true_obs):
  X_f = np.vstack([ens_T.reshape(n_ensemble, n_state_single).T,
                    ens_S.reshape(n_ensemble, n_state_single).T])
  Y_f = np.zeros((n_obs, n_ensemble))
  for k, (r, c) in enumerate(sensor_coords):
    Y_f[k, :] = ens_T[:, r, c]

  x_mean, y_mean = X_f.mean(1, keepdims=True), Y_f.mean(1, keepdims=True)
  X_p, Y_p = X_f - x_mean, Y_f - y_mean

  PfHt = (X_p @ Y_p.T) / (n_ensemble - 1) * loc_full
  HPfHt = (Y_p @ Y_p.T) / (n_ensemble - 1)
  K = PfHt @ np.linalg.inv(HPfHt + np.eye(n_obs) * R_var + np.eye(n_obs) * 1e-4)

  X_a = np.zeros_like(X_f)
  for j in range(n_ensemble):
    y_j = true_obs + np.random.normal(0, np.sqrt(R_var), size=n_obs)
    X_a[:, j] = X_f[:, j] + K @ (y_j - Y_f[:, j])

  std_f, std_a = X_f.std(1, keepdims=True), X_a.std(1, keepdims=True)
  x_a_mean = X_a.mean(1, keepdims=True)
  ratio = np.where(std_a > 1e-8, std_f / std_a, 1.0)
  X_a = x_a_mean + (X_a - x_a_mean) * (ALPHA_RTPS * ratio + (1 - ALPHA_RTPS))

  return (X_a[:n_state_single].T.reshape(n_ensemble, nx, ny),
          X_a[n_state_single:].T.reshape(n_ensemble, nx, ny))


true_field = np.zeros((nx, ny))
true_source = np.zeros((nx, ny))
ensemble_T = np.random.normal(0, 0.1, size=(n_ensemble, nx, ny))
ensemble_S = np.random.normal(0, 0.1, size=(n_ensemble, nx, ny))


# ============================================================================
# PARTE 3 -- Visualización (6 paneles, igual que osse_vs_enkf_animado.py)
# ============================================================================
fig, axes = plt.subplots(3, 2, figsize=(10, 13))

im1 = axes[0, 0].imshow(true_field, cmap='inferno', origin='lower', vmin=-2, vmax=2)
axes[0, 0].set_title('1. Temperatura Real (Oculta)')
for r, c in sensor_coords:
  axes[0, 0].plot(c, r, 'go', markersize=4, alpha=0.7)

im2 = axes[0, 1].imshow(true_source, cmap='plasma', origin='lower', vmin=-1, vmax=1)
axes[0, 1].set_title('2. Fuente Real (Móvil y Espacial)')

im3 = axes[1, 0].imshow(np.mean(ensemble_T, 0), cmap='inferno', origin='lower', vmin=-2, vmax=2)
axes[1, 0].set_title(f'3. Temperatura Estimada (EnKF, red óptima n={n_obs})')
for r, c in sensor_coords:
  axes[1, 0].plot(c, r, 'go', markersize=4, alpha=0.7)

im4 = axes[1, 1].imshow(np.mean(ensemble_S, 0), cmap='plasma', origin='lower', vmin=-1, vmax=1)
axes[1, 1].set_title('4. Fuente RECONSTRUIDA por el Filtro')

vmax_cmp = max(sigma_S_priori.max(), 0.05)
im5 = axes[2, 0].imshow(sigma_S_priori, cmap='viridis', origin='lower', vmin=0, vmax=vmax_cmp)
axes[2, 0].set_title('5. Error A PRIORI (Riccati) - Fuente')
for r, c in sensor_coords:
  axes[2, 0].plot(c, r, 'r.', markersize=4, alpha=0.7)

im6 = axes[2, 1].imshow(np.std(ensemble_S, 0), cmap='viridis', origin='lower', vmin=0, vmax=vmax_cmp)
axes[2, 1].set_title('6. Spread EMPÍRICO (EnKF real) - Fuente')
for r, c in sensor_coords:
  axes[2, 1].plot(c, r, 'r.', markersize=4, alpha=0.7)

for im, ax in [(im1, axes[0,0]), (im2, axes[0,1]), (im3, axes[1,0]),
               (im4, axes[1,1]), (im5, axes[2,0]), (im6, axes[2,1])]:
  fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

txt = fig.text(0.5, 0.005, '', ha='center', fontsize=10)


# ============================================================================
# PARTE 4 -- Bucle de animación
# ============================================================================
def animate(i):
  global true_field, true_source, ensemble_T, ensemble_S

  true_source = step_source(true_source)
  cx = int(nx / 2 + (nx * 0.35) * np.sin(i * 0.1))
  cy = int(ny / 2 + (ny * 0.35) * np.cos(i * 0.1))
  r_hot = max(2, nx // 12)
  true_source[max(0, cx-r_hot):min(nx, cx+r_hot), max(0, cy-r_hot):min(ny, cy+r_hot)] += 1.5

  true_field = step_physics(true_field, true_source)
  true_obs = np.array([true_field[r, c] for r, c in sensor_coords]) \
      + np.random.normal(0, np.sqrt(R_var), size=n_obs)

  for j in range(n_ensemble):
    ensemble_S[j] = step_source(ensemble_S[j])
    ensemble_T[j] = step_physics(ensemble_T[j], ensemble_S[j])

  ensemble_T, ensemble_S = enkf_step(ensemble_T, ensemble_S, true_obs)

  im1.set_array(true_field); im1.set_clim(true_field.min(), true_field.max())
  im2.set_array(true_source); im2.set_clim(true_source.min(), true_source.max())

  est_T = np.mean(ensemble_T, 0)
  im3.set_array(est_T); im3.set_clim(est_T.min(), est_T.max())

  est_S = np.mean(ensemble_S, 0)
  im4.set_array(est_S); im4.set_clim(est_S.min(), est_S.max())

  spread_S = np.std(ensemble_S, 0)
  im6.set_array(spread_S)

  txt.set_text(
      f"Frame {i}  |  RED ÓPTIMA (n={n_obs})  |  spread empírico (fuente) = {spread_S.mean():.4f}   "
      f"vs   a priori (Riccati) = {sigma_S_priori.mean():.4f}   "
      f"(ratio {spread_S.mean()/sigma_S_priori.mean():.2f}x)"
  )

  return [im1, im2, im3, im4, im6, txt]


ani = animation.FuncAnimation(fig, animate, frames=300, interval=50, blit=False)
plt.tight_layout(rect=[0, 0.02, 1, 1])
plt.show()