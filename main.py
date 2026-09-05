import time
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt

from src.grid import setup_grid, load_and_clean_data, build_observation_matrix_H
from src.enkf_jax import run_enkf_assimilation

def main():
    # 1. Configurar semilla determinista para JAX
    key = jax.random.PRNGKey(42)

    # 2. Configurar Malla 2D y Cargar Datos
    Nx, Ny = 25, 30
    x, y, X, Y, dx, dy = setup_grid(Nx=Nx, Ny=Ny)
    
    csv_path = 'data/Estaciones_PM2.5_2026_06.csv'
    Y_obs, timestamps, active_est_km, codes = load_and_clean_data(
        csv_path=csv_path, max_steps=96
    )
    
    # 3. Construir Operador de Observación H
    H = build_observation_matrix_H(active_est_km, x, y, Nx, Ny)
    
    print(f"Malla: {Nx}x{Ny} | Estaciones activas: {len(codes)} | Pasos de tiempo: {len(timestamps)}")
    
    # 4. Paso de calentamiento (JIT Compilation Warmup)
    print("Compilando grafos de cómputo JAX (Warmup)...")
    key, subkey = jax.random.split(key)
    _ = run_enkf_assimilation(
        subkey, Y_obs[:2], timestamps[:2], H, dx, dy, n_ensemble=10, Nx=Nx, Ny=Ny
    )
    
    # 5. Ejecución del filtro EnKF completo con benchmarking
    print("Ejecutando asimilación EnKF vectorizada en JAX...")
    start_time = time.time()
    
    key, subkey = jax.random.split(key)
    campo_reconstruido, ensemble_final = run_enkf_assimilation(
        subkey, Y_obs, timestamps, H, dx, dy, 
        dt=0.05, n_ensemble=40, Nx=Nx, Ny=Ny, R_std=2.5, Q_std=1.5
    )
    
    # Sincronizar llamadas asíncronas de JAX para medir tiempo real de ejecución
    campo_reconstruido.block_until_ready()
    elapsed_time = time.time() - start_time
    
    print(f"\n¡Asimilación completada con éxito!")
    print(f"Tiempo total de ejecución JAX: {elapsed_time:.4f} segundos")
    print(f"Velocidad promedio: {len(timestamps) / elapsed_time:.2f} pasos/segundo")

    # 6. Visualización del campo asimilado final
    plt.figure(figsize=(8, 6), dpi=120)
    plt.pcolormesh(X, Y, campo_reconstruido[-1], cmap='YlOrRd', shading='auto')
    plt.colorbar(label=r'PM2.5 ($\mu g / m^3$)')
    
    # Posiciones de las estaciones SIATA sobre la malla
    for code, (x_est, y_est) in active_est_km.items():
        plt.scatter(x_est, y_est, color='blue', edgecolors='white', zorder=5)
        plt.annotate(code, (x_est + 0.3, y_est + 0.3), fontsize=7, color='black', weight='bold')

    plt.title(f"Reconstrucción EnKF 2D con JAX (Hora t={len(timestamps)})", fontsize=11, fontweight='bold')
    plt.xlabel("X (km)")
    plt.ylabel("Y (km)")
    plt.grid(True, linestyle=':', alpha=0.4)
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    main()