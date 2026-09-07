import io
import os
from concurrent.futures import ProcessPoolExecutor
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from tqdm import tqdm

from src.physics_jax import get_wind_from_data

# Transformación aproximada de Grid Local (km) -> Lat/Lon (Valle de Aburrá)
LON_CENTER, LAT_CENTER = -75.575, 6.25

def km_to_latlon(x_km, y_km):
    lon = LON_CENTER + x_km / (111.32 * np.cos(np.radians(LAT_CENTER)))
    lat = LAT_CENTER + y_km / 110.57
    return lon, lat

def _render_single_frame(args):
    """Función trabajadora para renderizado en paralelo por CPU."""
    t, field, Lon, Lat, active_est_km, Y_obs_t, station_codes, timestamp, v_max = args
    
    fig, ax = plt.subplots(figsize=(8, 7.5), dpi=100)
    
    # 1. Contornos rellenados y líneas de nivel (PM2.5)
    levels = np.linspace(0, max(v_max, 40.0), 16)
    cf = ax.contourf(Lon, Lat, field, levels=levels, cmap='YlOrRd', extend='max')
    ax.contour(Lon, Lat, field, levels=levels, colors='brown', linewidths=0.3, alpha=0.4)
    
    # 2. Flecha de Viento Única en la Esquina Superior Derecha
    u_t, v_t = get_wind_from_data(t)
    v_mag = np.sqrt(u_t**2 + v_t**2)
    
    if v_mag > 1e-3:
        # Vector unitario para mantener la flecha con longitud constante en pantalla
        u_norm, v_norm = u_t / v_mag, v_t / v_mag
        
        ax.quiver(
            0.91, 0.90, u_norm, v_norm, transform=ax.transAxes,
            color='navy', scale=10, width=0.012, zorder=8
        )
        ax.text(
            0.91, 0.83, f"{v_mag:.1f} km/h", transform=ax.transAxes,
            ha='center', fontsize=8, fontweight='bold', color='navy',
            bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="navy", lw=0.8),
            zorder=8
        )

    # 3. Barra de color estilizada
    cbar = plt.colorbar(cf, ax=ax, pad=0.02, shrink=0.9, extendfrac=0.05)
    cbar.set_label(r'PM2.5 ($\mu g/m^3$)', fontsize=10, fontweight='bold')
    cbar.ax.axhline(13.0, color='green', linestyle='--', linewidth=1.5)
    cbar.ax.axhline(23.0, color='orange', linestyle='--', linewidth=1.5)

    # 4. Estaciones de Monitoreo
    for idx, (code, (x_est, y_est)) in enumerate(active_est_km.items()):
        lon_est, lat_est = km_to_latlon(x_est, y_est)
        val_obs = Y_obs_t[idx] if Y_obs_t is not None else 0.0
        
        dot_color = '#2ca02c' if val_obs < 15 else ('#ff7f0e' if val_obs < 35 else '#d62728')
        ax.scatter(lon_est, lat_est, c=dot_color, edgecolors='black', s=55, zorder=6)
        
        label_text = f"{code}\n({val_obs:.1f})"
        ax.annotate(
            label_text, 
            (lon_est, lat_est),
            xytext=(0, 12), textcoords="offset points",
            ha='center', fontsize=7, fontweight='bold',
            bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="black", lw=0.8),
            zorder=7
        )

    # 5. Título y Formato de Ejes
    ax.set_title(
        f"Digital Twin: Asimilación EnKF 2D - PM2.5\nValle de Aburrá | {timestamp.strftime('%Y-%m-%d %H:%M')}",
        fontsize=10, fontweight='bold', pad=10
    )
    ax.set_xlabel(r"Longitud ($^\circ$W)", fontsize=9, fontweight='bold')
    ax.set_ylabel(r"Latitud ($^\circ$N)", fontsize=9, fontweight='bold')
    
    ax.tick_params(labelsize=8)
    ax.set_xlim(Lon.min(), Lon.max())
    ax.set_ylim(Lat.min(), Lat.max())
    plt.tight_layout()

    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=100)
    plt.close(fig)
    buf.seek(0)
    return t, Image.open(buf).convert('RGB')

def generate_assimilation_gif(
    campo_reconstruido, X, Y, active_est_km, timestamps, Y_obs=None, station_codes=None, output_gif='aburra_pm25_enkf.gif'
):
    """Genera la animación GIF súper rápida procesando cuadros en paralelo."""
    total_steps = len(timestamps)
    v_max = float(np.max(campo_reconstruido))
    
    Lon, Lat = km_to_latlon(X, Y)
    
    tasks = [
        (t, campo_reconstruido[t], Lon, Lat, active_est_km, 
         Y_obs[t] if Y_obs is not None else None, station_codes, timestamps[t], v_max)
        for t in range(total_steps)
    ]

    print(f"\n[GIF] Renderizando {total_steps} cuadros en paralelo (Multiprocessing)...")
    
    frames_dict = {}
    max_workers = min(os.cpu_count() or 4, 8)
    
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        results = list(tqdm(executor.map(_render_single_frame, tasks), total=total_steps, desc="Renderizando"))
        for t, img in results:
            frames_dict[t] = img

    ordered_images = [frames_dict[t] for t in range(total_steps)]

    ordered_images[0].save(
        output_gif,
        save_all=True,
        append_images=ordered_images[1:],
        duration=120,
        loop=0
    )
    print(f"¡GIF guardado en estilo SIG profesional como '{output_gif}'!")