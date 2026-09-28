import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ==========================================
# PARÁMETROS DEL MURO Y GRÁFICO
# ==========================================
MURO_DIP = 70.0             # Inclinación del muro (°)
MURO_DIP_DIRECTION = 90.0   # Dirección de inclinación del muro (°)
ANGULO_FRICCION = 32.0      # Ángulo de fricción básica (°)
LIMITE_LATERAL = 20.0       # Límite lateral de Markland (°)

def obtener_ruta_csv(nombre_archivo="DIPS_MURO_FINAL.csv"):
    if os.path.exists(nombre_archivo):
        return nombre_archivo
    ruta_script = os.path.dirname(os.path.abspath(__file__))
    ruta_junta = os.path.join(ruta_script, nombre_archivo)
    if os.path.exists(ruta_junta):
        return ruta_junta
    raise FileNotFoundError(f"No se encontró '{nombre_archivo}'.")

def main():
    ruta = obtener_ruta_csv("DIPS_MURO_FINAL.csv")
    df = pd.read_csv(ruta, sep=";")
    df.columns = df.columns.str.strip()

    dips = df["Dip"].to_numpy(dtype=float)
    dip_dirs = df["Dip direction"].to_numpy(dtype=float)
    total_juntas = len(dips)

    # Convertir polos a proyección de igual área (hemisferio inferior)
    pole_trend = (dip_dirs + 180) % 360
    r_polos = np.sqrt(2) * np.sin(np.radians(dips) / 2.0)
    theta_polos = np.radians(pole_trend)

    # Identificar polos críticos por Deslizamiento Planar
    diff_dir = np.abs((dip_dirs - MURO_DIP_DIRECTION + 180) % 360 - 180)
    crit_planar = (diff_dir <= LIMITE_LATERAL) & (dips < MURO_DIP) & (dips > ANGULO_FRICCION)

    # Configuración de figura polar (estereograma)
    fig, ax = plt.subplots(figsize=(9, 9), subplot_kw={'projection': 'polar'})
    ax.set_theta_zero_location('N')  # Norte arriba
    ax.set_theta_direction(-1)       # Sentido horario

    # 1. Graficar Polos Estables
    ax.scatter(
        theta_polos[~crit_planar], 
        r_polos[~crit_planar], 
        c='navy', alpha=0.4, s=20, 
        label=f'Polos Estables ({np.sum(~crit_planar)})'
    )

    # 2. Graficar Polos Críticos (Deslizamiento Planar)
    ax.scatter(
        theta_polos[crit_planar], 
        r_polos[crit_planar], 
        c='red', alpha=0.8, s=35, 
        label=f'Polos Críticos Planar ({np.sum(crit_planar)})'
    )

    # 3. Graficar Polo del Muro
    muro_pole_trend = (MURO_DIP_DIRECTION + 180) % 360
    muro_r = np.sqrt(2) * np.sin(np.radians(MURO_DIP) / 2.0)
    ax.scatter(
        [np.radians(muro_pole_trend)], [muro_r], 
        c='darkgreen', s=120, marker='^', zorder=5, 
        label=f'Polo del Muro ({MURO_DIP}°/{MURO_DIP_DIRECTION}°)'
    )

    # 4. Graficar Gran Círculo (Plano del Muro)
    alphas = np.linspace(-np.pi/2, np.pi/2, 200)
    x_plane = np.cos(alphas)
    y_plane = np.sin(alphas) * np.cos(np.radians(MURO_DIP))
    z_plane = -np.sin(alphas) * np.sin(np.radians(MURO_DIP))

    plunge_p = np.degrees(np.arcsin(np.abs(z_plane)))
    trend_p = (MURO_DIP_DIRECTION - 90 + np.degrees(np.arctan2(x_plane, y_plane)) + 360) % 360
    r_p = np.sqrt(2) * np.sin(np.radians(90 - plunge_p) / 2.0)

    ax.plot(np.radians(trend_p), r_p, color='darkgreen', linewidth=2.5, label='Plano del Muro')

    # Ajustes finales del gráfico
    ax.set_ylim(0, np.sqrt(2))
    ax.set_yticks([])
    ax.set_title(
        f"Estereograma de Estabilidad Cinemática del Muro\nProyección de Igual Área (Hemisferio Inferior) - N = {total_juntas}", 
        pad=25, fontsize=12, fontweight='bold'
    )
    ax.legend(loc='lower right', frameon=True, facecolor='white', framealpha=0.9)

    plt.tight_layout()
    
    # Guardar imagen en disco
    nombre_imagen = "estereograma_muro_estabilidad.png"
    plt.savefig(nombre_imagen, dpi=300)
    print(f"Gráfico guardado exitosamente como '{nombre_imagen}'.")
    plt.show()

if __name__ == "__main__":
    main()