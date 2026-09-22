"""
Colocación óptima de sensores (a priori, greedy) usando el Riccati/OSSE
========================================================================
Responde la pregunta: "tengo presupuesto para N sensores, ¿dónde los pongo
para minimizar el error de reconstrucción de la fuente?" -- SIN simular ni
medir nada, solo con la dinámica del modelo (M, Q) y el ruido de sensor (R).

Idea del algoritmo (greedy / voraz):
  1. Se calcula la covarianza de pronóstico "climatológica": la incertidumbre
     en régimen estacionario si NO hubiera ningún sensor (solo el modelo
     evolucionando con su ruido de proceso). Es el punto de partida: cuánto
     no sabés antes de observar nada.
  2. Se agregan sensores de a uno. Para cada ubicación candidata (cada celda
     del grid) se calcula, con la fórmula de Sherman-Morrison, cuánto bajaría
     la incertidumbre total de la FUENTE si pusieras un sensor ahí -- sin
     tener que rehacer el Riccati completo cada vez (es una actualización de
     rango 1, muy barata).
  3. Se elige el candidato que más reduce la incertidumbre, se "instala" ahí
     (se actualiza la covarianza), y se repite hasta llegar a N sensores.

Esto es la aproximación estándar de diseño de redes de observación (parecida
a colocación óptima de sensores ambientales): es una aproximación "estática"
(no repite el ciclo completo de pronóstico-análisis paso a paso), pero en la
práctica da muy buenos resultados y es muchísimo más rápida que probar
combinaciones de sensores por fuerza bruta.

Al final, el script valida el resultado corriendo el Riccati COMPLETO
(cíclico, el mismo de osse_vs_enkf_animado.py) con la red de sensores
elegida, y lo compara contra una red uniforme con la misma cantidad de
sensores -- para confirmar que la red "optimizada" es realmente mejor.
"""

import time
import numpy as np
import matplotlib
matplotlib.use('Agg')  # solo genera imágenes, no abre ventana
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter

# ================== CONFIGURACIÓN ==================
nx, ny = 20, 20          # grid (mismo tamaño que las pruebas OSSE anteriores)
dt, D, DECAY = 0.1, 1.0, 0.02
n = nx * ny
n_state = 2 * n           # estado ampliado [Temperatura, Fuente]

R_var = 0.02               # varianza del ruido de cada sensor
N_SENSORS = 49              # presupuesto de sensores a colocar
CANDIDATOS_MARGEN = 1       # no considerar el borde exterior como candidato

np.random.seed(42)


def idx(r, c):
  return r * ny + c


# ============================================================================
# 1. Construir M (dinámica) y Q (ruido de proceso) -- igual que en el OSSE
# ============================================================================
print("Construyendo M y Q...")
t0 = time.time()

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
Q = np.zeros((n_state, n_state))
Q[n:, n:] = Q_S

print(f"  Listo en {time.time()-t0:.1f}s.")


# ============================================================================
# 2. Covarianza "climatológica": incertidumbre en régimen estacionario
#    SIN NINGÚN SENSOR (punto de partida del greedy).
# ============================================================================
print("Calculando covarianza climatológica (sin sensores)...")
t0 = time.time()
P_clima = np.eye(n_state) * 1.0
for it in range(150):
  P_clima = M @ P_clima @ M.T + Q
print(f"  Listo en {time.time()-t0:.1f}s. "
      f"Sigma media inicial (fuente) = {np.sqrt(np.diag(P_clima)[n:]).mean():.4f}")


# ============================================================================
# 3. Selección GREEDY de sensores (actualización de rango 1 / Sherman-Morrison)
# ============================================================================
def indices_candidatos():
  idxs = []
  for r in range(CANDIDATOS_MARGEN, nx - CANDIDATOS_MARGEN):
    for c in range(CANDIDATOS_MARGEN, ny - CANDIDATOS_MARGEN):
      idxs.append(idx(r, c))
  return np.array(idxs)


def greedy_seleccion(P_inicial, n_sensores, objetivo='fuente'):
  """
  objetivo: 'fuente'  -> minimiza la incertidumbre total SOLO de la fuente
            'total'    -> minimiza la incertidumbre total (T + fuente)
  Devuelve la lista de posiciones (r,c) elegidas, en orden de selección,
  y el historial del error total de la fuente después de cada sensor agregado.
  """
  P = P_inicial.copy()
  candidatos = indices_candidatos()
  elegidos = []
  historial_error_fuente = [np.sqrt(np.diag(P)[n:]).mean()]

  if objetivo == 'fuente':
    filas_objetivo = slice(n, n_state)
  else:
    filas_objetivo = slice(0, n_state)

  for paso in range(n_sensores):
    # Para cada candidato k (índice en el bloque T), calcular cuánto bajaría
    # la suma de varianzas del bloque objetivo si se agrega un sensor ahí.
    P_cols = P[filas_objetivo, :][:, candidatos]          # (n_obj, n_cand)
    numerador = np.sum(P_cols ** 2, axis=0)                 # (n_cand,)
    denom = R_var + P[candidatos, candidatos]                # (n_cand,)
    ganancia = numerador / denom                              # reducción esperada

    mejor = np.argmax(ganancia)
    k_star = candidatos[mejor]

    # Actualizar P con Sherman-Morrison (agregar el sensor elegido)
    col = P[:, k_star]
    denom_star = R_var + P[k_star, k_star]
    P = P - np.outer(col, col) / denom_star

    r_sel, c_sel = k_star // ny, k_star % ny
    elegidos.append((r_sel, c_sel))
    candidatos = np.delete(candidatos, mejor)  # no repetir ubicación

    historial_error_fuente.append(np.sqrt(np.diag(P)[n:]).mean())

  return elegidos, P, historial_error_fuente


print(f"Corriendo selección greedy para {N_SENSORS} sensores...")
t0 = time.time()
sensores_optimos, P_final_greedy, historial = greedy_seleccion(
    P_clima, N_SENSORS, objetivo='fuente')
print(f"  Listo en {time.time()-t0:.1f}s.")
print(f"  Sigma media final (fuente) tras {N_SENSORS} sensores óptimos: "
      f"{historial[-1]:.4f}  (arrancó en {historial[0]:.4f})")


# ============================================================================
# 4. Comparación: red óptima vs. red uniforme con la MISMA cantidad de sensores
# ============================================================================
def red_uniforme(n_sensores):
  lado = int(round(np.sqrt(n_sensores)))
  margen = 2
  coords_1d = np.linspace(margen, nx - 1 - margen, lado).round().astype(int)
  coords = [(int(r), int(c)) for r in coords_1d for c in coords_1d]
  return coords[:n_sensores]


def riccati_completo(sensor_coords, n_iter=150):
  """Riccati cíclico completo (el mismo que en osse_vs_enkf_animado.py),
  para VALIDAR el resultado del greedy con el cálculo 'de verdad'."""
  n_obs = len(sensor_coords)
  H = np.zeros((n_obs, n_state))
  for k, (r, c) in enumerate(sensor_coords):
    H[k, idx(r, c)] = 1.0
  R = np.eye(n_obs) * R_var

  P = np.eye(n_state) * 1.0
  for it in range(n_iter):
    P_f = M @ P @ M.T + Q
    S = H @ P_f @ H.T + R
    K = P_f @ H.T @ np.linalg.inv(S)
    P = P_f - K @ H @ P_f

  DFS = np.trace(K @ H)
  sigma_S = np.sqrt(np.diag(P)[n:]).reshape(nx, ny)
  sigma_T = np.sqrt(np.diag(P)[:n]).reshape(nx, ny)
  return sigma_T, sigma_S, DFS


coords_uniforme = red_uniforme(N_SENSORS)

print("\nValidando con el Riccati completo (cíclico)...")
sigma_T_opt, sigma_S_opt, DFS_opt = riccati_completo(sensores_optimos)
sigma_T_uni, sigma_S_uni, DFS_uni = riccati_completo(coords_uniforme)

print(f"\n=== RED ÓPTIMA (greedy, {N_SENSORS} sensores) ===")
print(f"  DFS = {DFS_opt:.2f}")
print(f"  Error fuente: media={sigma_S_opt.mean():.4f}, max={sigma_S_opt.max():.4f}")

print(f"\n=== RED UNIFORME ({len(coords_uniforme)} sensores) ===")
print(f"  DFS = {DFS_uni:.2f}")
print(f"  Error fuente: media={sigma_S_uni.mean():.4f}, max={sigma_S_uni.max():.4f}")

mejora = (sigma_S_uni.mean() - sigma_S_opt.mean()) / sigma_S_uni.mean() * 100
print(f"\n>>> La red óptima reduce el error medio de la fuente un {mejora:.1f}% "
      f"respecto a la red uniforme, con la MISMA cantidad de sensores. <<<")


# ============================================================================
# 5. Visualización
# ============================================================================
fig, axes = plt.subplots(2, 3, figsize=(16, 10))

vmax_sigma = max(sigma_S_opt.max(), sigma_S_uni.max())

ax = axes[0, 0]
im = ax.imshow(np.sqrt(np.diag(P_clima)[n:]).reshape(nx, ny), cmap='viridis',
               origin='lower', vmin=0, vmax=vmax_sigma)
ax.set_title('Incertidumbre SIN sensores\n(climatológica, punto de partida)')
fig.colorbar(im, ax=ax, fraction=0.046)

ax = axes[0, 1]
im = ax.imshow(sigma_S_opt, cmap='viridis', origin='lower', vmin=0, vmax=vmax_sigma)
rs, cs = zip(*sensores_optimos)
ax.plot(cs, rs, 'r.', markersize=6)
ax.set_title(f'Error a priori - RED ÓPTIMA (greedy)\nDFS={DFS_opt:.1f}, '
             f'media={sigma_S_opt.mean():.4f}')
fig.colorbar(im, ax=ax, fraction=0.046)

ax = axes[0, 2]
im = ax.imshow(sigma_S_uni, cmap='viridis', origin='lower', vmin=0, vmax=vmax_sigma)
ru, cu = zip(*coords_uniforme)
ax.plot(cu, ru, 'r.', markersize=6)
ax.set_title(f'Error a priori - RED UNIFORME\nDFS={DFS_uni:.1f}, '
             f'media={sigma_S_uni.mean():.4f}')
fig.colorbar(im, ax=ax, fraction=0.046)

ax = axes[1, 0]
ax.plot(historial, marker='o', markersize=3)
ax.set_xlabel('Sensores agregados (en orden)')
ax.set_ylabel('Sigma media de la fuente')
ax.set_title('Curva de mejora del greedy\n(rendimientos decrecientes)')
ax.grid(alpha=0.3)

ax = axes[1, 1]
ax.imshow(np.zeros((nx, ny)), cmap='gray', origin='lower', vmin=0, vmax=1)
for i, (r, c) in enumerate(sensores_optimos):
  ax.plot(c, r, 'yo', markersize=8)
  ax.annotate(str(i + 1), (c, r), color='black', ha='center', va='center', fontsize=6)
ax.set_title(f'Orden de colocación óptima\n(1 = más informativo, {N_SENSORS} = último agregado)')

ax = axes[1, 2]
ax.axis('off')
resumen = (
    f"RESUMEN\n\n"
    f"N sensores: {N_SENSORS}\n\n"
    f"Red óptima (greedy):\n"
    f"  DFS = {DFS_opt:.2f}\n"
    f"  error medio fuente = {sigma_S_opt.mean():.4f}\n\n"
    f"Red uniforme:\n"
    f"  DFS = {DFS_uni:.2f}\n"
    f"  error medio fuente = {sigma_S_uni.mean():.4f}\n\n"
    f"Mejora: {mejora:.1f}%"
)
ax.text(0.05, 0.5, resumen, fontsize=11, va='center', family='monospace')

plt.tight_layout()
plt.savefig('sensores_optimos.png', dpi=120)
print("\nGráfico guardado en sensores_optimos.png")

# Guardar las coordenadas óptimas para poder reusarlas en otro script
np.save('sensores_optimos_coords.npy', np.array(sensores_optimos))
print("Coordenadas guardadas en sensores_optimos_coords.npy "
      "(array de forma (N_SENSORS, 2), cada fila es (r, c))")

np.savez('sensores_optimos.npz',
         coords=np.array(sensores_optimos), nx=nx, ny=ny)