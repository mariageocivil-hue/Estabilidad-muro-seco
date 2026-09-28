import os
import pandas as pd
import numpy as np

# ==========================================
# PARÁMETROS DEL MURO Y ANÁLISIS CINEMÁTICO
# ==========================================
COL_DIP = "Dip"
COL_DIP_DIRECTION = "Dip direction"

MURO_DIP = 70.0             # Inclinación del muro/talud (°)
MURO_DIP_DIRECTION = 90.0   # Dirección de inclinación del muro (°)
ANGULO_FRICCION = 32.0      # Ángulo de fricción básica (°)
LIMITE_LATERAL = 20.0       # Límite lateral de Markland (°)

def obtener_ruta_csv(nombre_archivo="DIPS_MURO_FINAL.csv"):
    """Busca el archivo CSV en la carpeta actual o en la carpeta donde está guardado el script."""
    # 1. Probar en la carpeta actual de trabajo
    if os.path.exists(nombre_archivo):
        return nombre_archivo
    
    # 2. Probar en la misma carpeta del script .py
    ruta_script = os.path.dirname(os.path.abspath(__file__))
    ruta_junta = os.path.join(ruta_script, nombre_archivo)
    if os.path.exists(ruta_junta):
        return ruta_junta

    raise FileNotFoundError(f"No se encontró el archivo '{nombre_archivo}'. Asegúrate de que esté en la misma carpeta que el código.")

def cargar_y_limpiar_csv(nombre_archivo="DIPS_MURO_FINAL.csv"):
    """Carga y limpia el archivo CSV de discontinuidades."""
    ruta = obtener_ruta_csv(nombre_archivo)
    df = pd.read_csv(ruta, sep=";")
    df.columns = df.columns.str.strip()
    return df

def main():
    print("Cargando datos de discontinuidades...")
    try:
        df = cargar_y_limpiar_csv("DIPS_MURO_FINAL.csv")
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return

    total_juntas = len(df)
    dips = df[COL_DIP].to_numpy(dtype=float)
    dip_dirs = df[COL_DIP_DIRECTION].to_numpy(dtype=float)

    print(f"Procesando {total_juntas} mediciones de discontinuidades...")

    # 1. DESLIZAMIENTO PLANAR (Planar Sliding)
    diff_dir = np.abs((dip_dirs - MURO_DIP_DIRECTION + 180) % 360 - 180)
    crit_planar = (diff_dir <= LIMITE_LATERAL) & (dips < MURO_DIP) & (dips > ANGULO_FRICCION)
    n_planar = int(np.sum(crit_planar))

    # 2. VOLCAMIENTO FLEXURAL (Flexural Toppling)
    diff_dir_opp = np.abs((dip_dirs - (MURO_DIP_DIRECTION + 180) % 360 + 180) % 360 - 180)
    crit_flexural = (diff_dir_opp <= LIMITE_LATERAL) & (dips >= (90 - MURO_DIP + ANGULO_FRICCION))
    n_flexural = int(np.sum(crit_flexural))

    # 3. DESLIZAMIENTO EN CUÑA (Wedge Sliding)
    dip_rad = np.radians(dips)
    dd_rad = np.radians(dip_dirs)
    nx = np.sin(dip_rad) * np.sin(dd_rad)
    ny = np.sin(dip_rad) * np.cos(dd_rad)
    nz = -np.cos(dip_rad)
    poles = np.column_stack([nx, ny, nz])

    i_idx, j_idx = np.triu_indices(total_juntas, k=1)
    cross = np.cross(poles[i_idx], poles[j_idx])
    norms = np.linalg.norm(cross, axis=1)

    valid = norms > 1e-5
    cross = cross[valid]
    norms = norms[valid]
    cross = cross / norms[:, None]
    cross[cross[:, 2] > 0] *= -1

    plunge = np.degrees(np.arcsin(np.abs(cross[:, 2])))
    trend = (np.degrees(np.arctan2(cross[:, 0], cross[:, 1])) + 360) % 360

    diff_trend = np.abs((trend - MURO_DIP_DIRECTION + 180) % 360 - 180)
    crit_wedge = (diff_trend <= 90) & (plunge < MURO_DIP) & (plunge > ANGULO_FRICCION)
    n_wedge_intersections = int(np.sum(crit_wedge))
    total_intersecciones = len(cross)

    pct_planar = (n_planar / total_juntas) * 100
    pct_flexural = (n_flexural / total_juntas) * 100
    pct_wedge = (n_wedge_intersections / total_intersecciones) * 100

    # 4. IMPRESIÓN Y GUARDADO DEL REPORTE
    reporte = f"""============================================
REPORTE DE ESTABILIDAD CINEMÁTICA EN MUROS
============================================

Proyecto: Fragmento de muro de piedra seca
Total de discontinuidades analizadas: {total_juntas}
Paramento del muro: Dip = {MURO_DIP}°, Dip Direction = {MURO_DIP_DIRECTION}°
Ángulo de fricción asumido: {ANGULO_FRICCION}°
Límite lateral (Markland): {LIMITE_LATERAL}°

--------------------------------------------
RESUMEN DE RESULTADOS CINEMÁTICOS
--------------------------------------------
1. Deslizamiento Planar (Planar Sliding):
   - Juntas críticas: {n_planar} / {total_juntas} ({pct_planar:.2f}%)

2. Volcamiento Flexural (Flexural Toppling):
   - Juntas críticas: {n_flexural} / {total_juntas} ({pct_flexural:.2f}%)

3. Deslizamiento en Cuña (Wedge Sliding):
   - Intersecciones críticas: {n_wedge_intersections} / {total_intersecciones} ({pct_wedge:.2f}%)

--------------------------------------------
CRITERIO DE INTERPRETACIÓN
--------------------------------------------
  < 10%  -> Nivel de inestabilidad bajo
  10-30% -> Nivel de inestabilidad moderado
  > 30%  -> Nivel de inestabilidad elevado
"""

    with open("reporte_estabilidad.rst", "w", encoding="utf-8") as f:
        f.write(reporte)

    print("\n" + reporte)
    print("Reporte generado y guardado exitosamente en 'reporte_estabilidad.rst'.")

if __name__ == "__main__":
    main()