# ============================================================
# PROYECTO: VELOX UTVT
# PRACTICA 20 (ETAPA 12): GENERADOR DE DATOS MASIVOS (BIG DATA)
#
# Archivo:
# p20generador_datos_masivos_velox.py
#
# Objetivo:
# Calibrar parámetros estadísticos a partir de los viajes reales
# almacenados en MongoDB Atlas ('db.viaje') y escalar el volumen
# a 50,000 registros estructurados en bloques vectorizados para
# alimentar las pipelines de Big Data (Parquet, Apache Spark y MLP).
#
# Ejecución desde terminal:
# python p20generador_datos_masivos_velox.py
# ============================================================

import os
import sys
import time
from pathlib import Path
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

# Permitir importar 'database.py' desde la raíz del proyecto Velox UTVT
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from database import db

# ============================================================
# 1. CONFIGURACION GENERAL
# ============================================================

SEMILLA = 42
TOTAL_REGISTROS = 50_000
TAMANO_LOTE = 10_000
ESTADOS_EXITOSOS = ["activo", "aceptado", "en_curso", "finalizado"]

RUTA_ACTUAL = Path(__file__).resolve().parent
CARPETA_BIGDATA = RUTA_ACTUAL / "resultados_bigdata_velox"
CARPETA_BIGDATA.mkdir(parents=True, exist_ok=True)
ARCHIVO_SALIDA = CARPETA_BIGDATA / "velox_viajes_masivos_50k.csv"

RUTAS_UTVT = [
    ("Terminal Toluca", 22.5, 18.0),
    ("Metepec Centro", 19.0, 16.0),
    ("Lerma Centro", 9.5, 12.0),
    ("San Mateo Atenco", 13.0, 14.0),
    ("Toluca Centro", 24.0, 20.0),
    ("Xonacatlán", 16.5, 15.0),
    ("La Zapata / Toluca-Naucalpan", 15.0, 14.0)
]

HORARIOS_UNIVERSITARIOS = [
    "06:30", "07:00", "07:30", "08:00",
    "13:00", "14:00", "15:00", "18:00", "19:30"
]


# ============================================================
# 2. EXTRAER PERFIL ESTADISTICO REAL DE MONGODB ATLAS
# ============================================================

def obtener_parametros_reales_mongodb():
    """
    Consulta 'db.viaje' en MongoDB Atlas para extraer las métricas reales
    de Velox UTVT y usarlas como línea base de calibración.
    """
    print("\n1. CONSULTANDO LINEA BASE REAL EN MONGODB ATLAS (db.viaje)")
    print("-" * 65)

    registros_reales = []
    try:
        cursor = db.viaje.find({}, {"_id": 0, "costo": 1, "asientos_disponibles": 1, "estado": 1})
        for doc in cursor:
            estado = str(doc.get("estado", "")).lower().strip()
            registros_reales.append({
                "costo": float(doc.get("costo", 15.0)),
                "asientos_disponibles": int(doc.get("asientos_disponibles", 2)),
                "viaje_exitoso": 1 if estado in ESTADOS_EXITOSOS else 0
            })
    except Exception as error:
        print(f"Aviso: No se pudo conectar a Atlas ({error}). Se usarán parámetros base.")

    if registros_reales:
        df_real = pd.DataFrame(registros_reales)
        stats = {
            "n_reales": len(df_real),
            "costo_media": float(df_real["costo"].mean()),
            "costo_std": max(float(df_real["costo"].std(ddof=0)), 3.5),
            "costo_min": float(df_real["costo"].min()),
            "costo_max": float(df_real["costo"].max()),
            "asientos_media": float(df_real["asientos_disponibles"].mean()),
            "tasa_exito_real": float(df_real["viaje_exitoso"].mean())
        }
        print(f"Viajes reales encontrados en MongoDB: {stats['n_reales']}")
        print(f"Tarifa real promedio: ${stats['costo_media']:.2f} MXN (rango ${stats['costo_min']:.2f} - ${stats['costo_max']:.2f})")
        print(f"Asientos reales promedio: {stats['asientos_media']:.2f}")
        print(f"Tasa real de consolidación: {stats['tasa_exito_real'] * 100:.2f}%")
        return stats

    return {
        "n_reales": 0,
        "costo_media": 16.0,
        "costo_std": 5.0,
        "costo_min": 10.0,
        "costo_max": 35.0,
        "asientos_media": 2.5,
        "tasa_exito_real": 0.65
    }


# ============================================================
# 3. GENERADOR VECTORIZADO POR LOTES (50,000 REGISTROS)
# ============================================================

def generar_lote_viajes(rng, inicio_idx, cantidad, stats_reales):
    """
    Genera un bloque vectorizado de 'cantidad' viajes calibrado con
    las estadísticas reales de Velox UTVT y reglas operativas de negocio.
    """
    ids_viaje = [f"VLX-BD-{i:06d}" for i in range(inicio_idx, inicio_idx + cantidad)]
    ids_conductor = [f"2224{rng.integers(10000, 99999)}" for _ in range(cantidad)]

    # Selección de rutas reales del Valle de Toluca -> UTVT
    indices_ruta = rng.integers(0, len(RUTAS_UTVT), size=cantidad)
    origenes = [RUTAS_UTVT[i][0] for i in indices_ruta]
    distancias_km = np.array([RUTAS_UTVT[i][1] for i in indices_ruta], dtype=float)
    tarifas_base = np.array([RUTAS_UTVT[i][2] for i in indices_ruta], dtype=float)
    destinos = ["Campus UTVT"] * cantidad

    horarios = rng.choice(HORARIOS_UNIVERSITARIOS, size=cantidad)

    # Fechas distribuidas en los últimos 180 días
    dias_atras = rng.integers(0, 180, size=cantidad)
    fecha_base = datetime(2026, 11, 22)
    fechas = [(fecha_base - timedelta(days=int(d))).strftime("%Y-%m-%d") for d in dias_atras]

    # Costo calibrado alrededor de la tarifa de la ruta y la media real de MongoDB
    ajuste_real = (stats_reales["costo_media"] - 16.0) * 0.4
    ruido_tarifa = rng.normal(loc=ajuste_real, scale=stats_reales["costo_std"], size=cantidad)
    costo = np.clip(np.round(tarifas_base + ruido_tarifa, 2), 8.0, 65.0)

    # Asientos disponibles (1 a 5) y puntualidad/calificación del conductor
    asientos_disponibles = rng.integers(1, 6, size=cantidad)
    puntualidad = np.clip(np.round(rng.normal(86.0, 10.0, size=cantidad), 1), 40.0, 100.0)
    calificacion_conductor = np.clip(np.round(rng.normal(4.5, 0.45, size=cantidad), 2), 2.5, 5.0)
    viajes_previos = rng.integers(0, 85, size=cantidad)

    # Cálculo de probabilidad latente de éxito del viaje
    # Combina accesibilidad económica, asientos, puntualidad y reputación
    score_costo = (30.0 - costo) / 20.0               # Mayor si el costo es accesible
    score_asientos = (asientos_disponibles - 2.0) / 3.0
    score_puntualidad = (puntualidad - 75.0) / 25.0
    score_reputacion = (calificacion_conductor - 3.8) / 1.2

    logit = (
        0.45
        + 1.35 * score_costo
        + 0.55 * score_asientos
        + 0.90 * score_puntualidad
        + 0.75 * score_reputacion
        + rng.normal(0, 0.35, size=cantidad)  # Ruido realista
    )
    probabilidad_exito = 1.0 / (1.0 + np.exp(-logit))
    viaje_exitoso = (probabilidad_exito >= 0.52).astype(int)

    # Asignar estado operativo consistente con los estados reales de Velox UTVT
    estados = np.where(
        viaje_exitoso == 1,
        rng.choice(["aceptado", "finalizado", "activo"], size=cantidad, p=[0.45, 0.40, 0.15]),
        rng.choice(["cancelado", "pendiente"], size=cantidad, p=[0.60, 0.40])
    )

    return pd.DataFrame({
        "id_viaje": ids_viaje,
        "id_conductor": ids_conductor,
        "fecha": fechas,
        "horario": horarios,
        "origen": origenes,
        "destino": destinos,
        "distancia_km": distancias_km,
        "costo": costo,
        "asientos_disponibles": asientos_disponibles,
        "puntualidad": puntualidad,
        "calificacion_conductor": calificacion_conductor,
        "viajes_previos": viajes_previos,
        "estado": estados,
        "viaje_exitoso": viaje_exitoso
    })


# ============================================================
# 4. PIPELINE PRINCIPAL DE GENERACION Y EXPORTACION
# ============================================================

def main():
    print("=" * 65)
    print("VELOX UTVT - PRACTICA 20 (ETAPA 12)")
    print("GENERACION DE DATASET MASIVO PARA BIG DATA (50,000 REGISTROS)")
    print("=" * 65)

    inicio = time.perf_counter()
    rng = np.random.default_rng(SEMILLA)

    # Paso 1: Extraer línea base real desde MongoDB Atlas
    stats_reales = obtener_parametros_reales_mongodb()

    # Paso 2: Generar por lotes y escribir incrementalmente en CSV
    print("\n2. GENERANDO DATASET MASIVO POR LOTES")
    print("-" * 65)

    if ARCHIVO_SALIDA.exists():
        ARCHIVO_SALIDA.unlink()

    total_generados = 0
    total_exitosos = 0
    suma_costos = 0.0

    for offset in range(0, TOTAL_REGISTROS, TAMANO_LOTE):
        lote_actual = min(TAMANO_LOTE, TOTAL_REGISTROS - offset)
        df_lote = generar_lote_viajes(rng, offset + 1, lote_actual, stats_reales)

        escribir_encabezado = (offset == 0)
        df_lote.to_csv(
            ARCHIVO_SALIDA,
            mode="a",
            header=escribir_encabezado,
            index=False,
            encoding="utf-8"
        )

        total_generados += len(df_lote)
        total_exitosos += int(df_lote["viaje_exitoso"].sum())
        suma_costos += float(df_lote["costo"].sum())
        print(f"  Lote procesado: {total_generados:,} / {TOTAL_REGISTROS:,} registros...")

    duracion = time.perf_counter() - inicio
    tamano_mb = ARCHIVO_SALIDA.stat().st_size / (1024 * 1024)

    print("\n3. RESUMEN DE ESCALAMIENTO BIG DATA")
    print("-" * 65)
    print(f"Archivo generado: {ARCHIVO_SALIDA}")
    print(f"Total de registros: {total_generados:,}")
    print(f"Tamaño en disco: {tamano_mb:.2f} MB")
    print(f"Tarifa promedio generada: ${suma_costos / total_generados:.2f} MXN")
    print(f"Balance de clase (viaje_exitoso=1): {(total_exitosos / total_generados) * 100:.2f}%")
    print(f"Tiempo total de generación: {duracion:.2f} segundos")
    print("\nPRACTICA 20 FINALIZADA CORRECTAMENTE.")


if __name__ == "__main__":
    main()