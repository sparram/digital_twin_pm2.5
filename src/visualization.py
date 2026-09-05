import io
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from tqdm import tqdm  # Barra de progreso
from src.physics_jax import get_wind_aburra

def generate_assimilation_gif(
    campo_reconstruido, X, Y, active_est_km, timestamps, output_gif='aburra_pm25_enkf.gif'
):
    """Genera una animación GIF con la evolución diurna del campo asimilado y el viento."""
    images = []
    total_steps = len(timestamps)
    
    v_max = float(np.max(campo_reconstruido))
    v_min = float(np.min(campo_reconstruido))
    
    print(f"\n[GIF] Generando animación ({total_steps} cuadros)...")
    
    # Bucle con barra de progreso tqdm
    for t in tqdm(range(total_steps), desc="Renderizando cuadros del GIF"):
        fig, ax = plt.subplots(figsize=(7.5, 6.5), dpi=100)
        
        # 1. Campo escalar 2D de PM2.5
        mesh = ax.pcolormesh(X, Y, campo_reconstruido[t], cmap='YlOrRd', vmin=v_min, vmax=v_max, shading='auto')
        cbar = plt.colorbar(mesh, ax=ax, shrink=0.85)
        cbar.set_label(r'$PM_{2.5}$ ($\mu g / m^3$)', fontsize=9)
        
        # 2. Vector de viento (flecha indicativa)
        u_w, v_w = get_wind_aburra(timestamps[t].hour)
        ax.quiver(-7.5, 10, u_w, v_w, scale=18, color='blue', width=0.009, zorder=6)
        ax.text(-7.2, 11.2, f"Viento: u={u_w:.1f}, v={v_w:.1f} km/h", color='blue', fontsize=8, weight='bold')
        
        # 3. Estaciones de monitoreo SIATA
        for code, (x_est, y_est) in active_est_km.items():
            ax.scatter(x_est, y_est, color='blue', edgecolors='white', zorder=7, s=35)
            ax.text(x_est + 0.3, y_est + 0.3, code, fontsize=7, fontweight='bold', color='black')
            
        ax.set_title(f"Gemelo Digital Aburrá PM2.5 — {timestamps[t].strftime('%Y-%m-%d %H:%M')}", fontsize=10, fontweight='bold')
        ax.set_xlabel("X (km)", fontsize=9)
        ax.set_ylabel("Y (km)", fontsize=9)
        ax.set_xlim(X.min(), X.max())
        ax.set_ylim(Y.min(), Y.max())
        ax.grid(True, linestyle=':', alpha=0.3)
        plt.tight_layout()
        
        # Convertir figura a objeto de imagen PIL en memoria
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=100)
        plt.close(fig)
        buf.seek(0)
        images.append(Image.open(buf).convert('RGB'))
        
    # Guardar como GIF animado (8 FPS)
    images[0].save(
        output_gif,
        save_all=True,
        append_images=images[1:],
        duration=125,
        loop=0
    )
    print(f"¡GIF guardado exitosamente como '{output_gif}'!")