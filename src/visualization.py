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
    """Función trabajadora modificada para renderizar C y S lado a lado."""
    t, field_c, field_s, Lon, Lat, active_est_km, Y_obs_t, station_codes, timestamp, v_max_c, v_max_s = args
    
    # Creamos una figura con 2 subplots lado a lado
    fig, axes = plt.subplots(1, 2, figsize=(15, 7), dpi=100)
    ax_c, ax_s = axes
    
    # ==========================================
    # PANEL 1: CONCENTRACIÓN PM2.5 (C)
    # ==========================================
    levels_c = np.linspace(0, max(v_max_c, 40.0), 16)
    cf_c = ax_c.contourf(Lon, Lat, field_c, levels=levels_c, cmap='YlOrRd', extend='max')
    ax_c.contour(Lon, Lat, field_c, levels=levels_c, colors='brown', linewidths=0.3, alpha=0.4)
    
    # Viento en el panel de concentración
    u_t, v_t = get_wind_from_data(t)
    v_mag = np.sqrt(u_t**2 + v_t**2)
    if v_mag > 1e-3:
        u_norm, v_norm = u_t / v_mag, v_t / v_mag
        ax_c.quiver(0.91, 0.90, u_norm, v_norm, transform=ax_c.transAxes, color='navy', scale=10, width=0.012, zorder=8)
        ax_c.text(0.91, 0.83, f"{v_mag:.1f} km/h", transform=ax_c.transAxes, ha='center', fontsize=8, fontweight='bold', color='navy', bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="navy", lw=0.8), zorder=8)

    # Barra de color PM2.5
    cbar_c = plt.colorbar(cf_c, ax=ax_c, pad=0.02, shrink=0.9, extendfrac=0.05)
    cbar_c.set_label(r'PM2.5 ($\mu g/m^3$)', fontsize=10, fontweight='bold')
    cbar_c.ax.axhline(13.0, color='green', linestyle='--', linewidth=1.5)
    cbar_c.ax.axhline(23.0, color='orange', linestyle='--', linewidth=1.5)

    # Estaciones en panel C
    for idx, (code, (x_est, y_est)) in enumerate(active_est_km.items()):
        lon_est, lat_est = km_to_latlon(x_est, y_est)
        val_obs = Y_obs_t[idx] if Y_obs_t is not None else 0.0
        dot_color = '#2ca02c' if val_obs < 15 else ('#ff7f0e' if val_obs < 35 else '#d62728')
        ax_c.scatter(lon_est, lat_est, c=dot_color, edgecolors='black', s=55, zorder=6)
        label_text = f"{code}\n({val_obs:.1f})"
        ax_c.annotate(label_text, (lon_est, lat_est), xytext=(0, 12), textcoords="offset points", ha='center', fontsize=7, fontweight='bold', bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="black", lw=0.8), zorder=7)

    ax_c.set_title(f"Concentración PM2.5\nValle de Aburrá | {timestamp.strftime('%Y-%m-%d %H:%M')}", fontsize=10, fontweight='bold', pad=10)
    ax_c.set_xlabel(r"Longitud ($^\circ$W)", fontsize=9, fontweight='bold')
    ax_c.set_ylabel(r"Latitud ($^\circ$N)", fontsize=9, fontweight='bold')
    ax_c.tick_params(labelsize=8)
    ax_c.set_xlim(Lon.min(), Lon.max())
    ax_c.set_ylim(Lat.min(), Lat.max())

    # ==========================================
    # PANEL 2: FUENTE ESTIMADA (S)
    # ==========================================
    levels_s = np.linspace(0, max(v_max_s, 1e-3), 16)
    cf_s = ax_s.contourf(Lon, Lat, field_s, levels=levels_s, cmap='Purples', extend='max')
    ax_s.contour(Lon, Lat, field_s, levels=levels_s, colors='indigo', linewidths=0.3, alpha=0.4)

    # Barra de color Fuente S
    cbar_s = plt.colorbar(cf_s, ax=ax_s, pad=0.02, shrink=0.9, extendfrac=0.05)
    cbar_s.set_label(r'Tasa de Emisión S ($[units/s]$)', fontsize=10, fontweight='bold')

    # Opcional: Mostrar las estaciones también en el mapa de fuentes como referencia espacial
    for idx, (code, (x_est, y_est)) in enumerate(active_est_km.items()):
        lon_est, lat_est = km_to_latlon(x_est, y_est)
        ax_s.scatter(lon_est, lat_est, c='gray', edgecolors='black', s=30, alpha=0.6, zorder=6)

    ax_s.set_title(f"Fuente Estimada ($S$ - Estado Aumentado)\nValle de Aburrá | {timestamp.strftime('%Y-%m-%d %H:%M')}", fontsize=10, fontweight='bold', pad=10)
    ax_s.set_xlabel(r"Longitud ($^\circ$W)", fontsize=9, fontweight='bold')
    ax_s.set_ylabel(r"Latitud ($^\circ$N)", fontsize=9, fontweight='bold')
    ax_s.tick_params(labelsize=8)
    ax_s.set_xlim(Lon.min(), Lon.max())
    ax_s.set_ylim(Lat.min(), Lat.max())

    plt.tight_layout()

    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=100)
    plt.close(fig)
    buf.seek(0)
    return t, Image.open(buf).convert('RGB')

def generate_assimilation_gif(
    campo_reconstruido, campo_fuentes, X, Y, active_est_km, timestamps, Y_obs=None, station_codes=None, output_gif='aburra_pm25_enkf.gif'
):
    """Genera la animación GIF comparativa (Concentración vs Fuente) en paralelo."""
    total_steps = len(timestamps)
    v_max_c = float(np.max(campo_reconstruido))
    v_max_s = float(np.max(campo_fuentes)) if campo_fuentes is not None else 1.0
    
    Lon, Lat = km_to_latlon(X, Y)
    
    tasks = [
        (
            t, 
            campo_reconstruido[t], 
            campo_fuentes[t] if campo_fuentes is not None else np.zeros_like(campo_reconstruido[t]),
            Lon, Lat, active_est_km, 
            Y_obs[t] if Y_obs is not None else None, 
            station_codes, timestamps[t], v_max_c, v_max_s
        )
        for t in range(total_steps)
    ]

    print(f"\n[GIF Dual] Renderizando {total_steps} cuadros en paralelo (C y S)...")
    
    frames_dict = {}
    max_workers = min(os.cpu_count() or 4, 8)
    
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        results = list(tqdm(executor.map(_render_single_frame, tasks), total=total_steps, desc="Renderizando"))
        for t, img in results:
            frames_dict[t] = img

    ordered_images = [frames_dict[t] for t in range(total_steps)]

    os.makedirs("media", exist_ok=True)
    ordered_images[0].save(
        "media/" + output_gif,
        save_all=True,
        append_images=ordered_images[1:],
        duration=120,
        loop=0
    )
    print(f"¡GIF dual guardado exitosamente como 'media/{output_gif}'!")