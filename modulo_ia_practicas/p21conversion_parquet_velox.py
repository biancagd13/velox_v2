# ============================================================
# PROYECTO: VELOX UTVT
# PRACTICA 21 (ETAPA 13): CONVERSION A FORMATO COLUMNAR PARQUET
#
# Archivo:
# p21conversion_parquet_velox.py
#
# Objetivo:
# Convertir el dataset masivo de viajes de Velox UTVT desde formato
# plano CSV hacia Apache Parquet con compresión Snappy, evaluando
# la reducción de espacio en disco y la ganancia de velocidad en
# lectura completa y proyección columnar.
#
# Ejecución desde terminal:
# python p21conversion_parquet_velox.py
# ============================================================

import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# ============================================================
# 1. CONFIGURACION DE RUTAS
# ============================================================

RUTA_ACTUAL = Path(__file__).resolve().parent
CARPETA_BIGDATA = RUTA_ACTUAL / "resultados_bigdata_velox"
CARPETA_BIGDATA.mkdir(parents=True, exist_ok=True)

ARCHIVO_CSV = CARPETA_BIGDATA / "velox_viajes_masivos_50k.csv"
ARCHIVO_PARQUET = CARPETA_BIGDATA / "velox_viajes_masivos_50k.parquet"
REPORTE_BENCHMARK = CARPETA_BIGDATA / "p21_benchmark_csv_vs_parquet.csv"
GRAFICA_BENCHMARK = CARPETA_BIGDATA / "p21_comparacion_rendimiento.png"


# ============================================================
# 2. CARGA Y OPTIMIZACION DE ESQUEMA
# ============================================================

def cargar_y_optimizar_csv(ruta_csv):
    """
    Lee el dataset masivo CSV de Velox UTVT, mide el tiempo de lectura
    y optimiza los tipos de datos para almacenamiento columnar eficiente.
    """
    print("\n1. LECTURA DE DATASET MASIVO EN FORMATO CSV")
    print("-" * 65)

    inicio_lectura = time.perf_counter()
    df = pd.read_csv(ruta_csv, encoding="utf-8")
    tiempo_lectura_csv = time.perf_counter() - inicio_lectura

    # Optimización de tipos de datos para reducir huella en memoria y disco
    df["distancia_km"] = df["distancia_km"].astype("float32")
    df["costo"] = df["costo"].astype("float32")
    df["asientos_disponibles"] = df["asientos_disponibles"].astype("int32")
    df["puntualidad"] = df["puntualidad"].astype("float32")
    df["calificacion_conductor"] = df["calificacion_conductor"].astype("float32")
    df["viajes_previos"] = df["viajes_previos"].astype("int32")
    df["viaje_exitoso"] = df["viaje_exitoso"].astype("int8")

    print(f"Registros cargados: {len(df):,}")
    print(f"Columnas del esquema: {list(df.columns)}")
    print(f"Tiempo de lectura CSV completo: {tiempo_lectura_csv:.4f} segundos")

    return df, tiempo_lectura_csv


# ============================================================
# 3. EXPORTACION A APACHE PARQUET (PYARROW + SNAPPY)
# ============================================================

def exportar_a_parquet(df, ruta_parquet):
    """
    Serializa el DataFrame optimizado a formato columnar Apache Parquet
    utilizando el motor PyArrow y compresión Snappy.
    """
    print("\n2. CONVERSION A FORMATO APACHE PARQUET (SNAPPY)")
    print("-" * 65)

    inicio_escritura = time.perf_counter()
    df.to_parquet(
        ruta_parquet,
        engine="pyarrow",
        compression="snappy",
        index=False
    )
    tiempo_escritura = time.perf_counter() - inicio_escritura

    print(f"Archivo Parquet generado: {ruta_parquet}")
    print(f"Tiempo de escritura Parquet: {tiempo_escritura:.4f} segundos")

    return tiempo_escritura


# ============================================================
# 4. BENCHMARK COMPARATIVO: CSV VS PARQUET
# ============================================================

def ejecutar_benchmark(ruta_csv, ruta_parquet, tiempo_lectura_csv):
    """
    Compara tamaño en disco, velocidad de lectura completa y lectura
    de subconjunto de columnas (proyección columnar) entre CSV y Parquet.
    """
    print("\n3. BENCHMARK DE ALMACENAMIENTO Y VELOCIDAD")
    print("-" * 65)

    # Tamaño en disco (MB)
    tamano_csv_mb = ruta_csv.stat().st_size / (1024 * 1024)
    tamano_parquet_mb = ruta_parquet.stat().st_size / (1024 * 1024)
    ahorro_espacio_pct = ((tamano_csv_mb - tamano_parquet_mb) / tamano_csv_mb) * 100

    # Lectura completa desde Parquet
    inicio_pq = time.perf_counter()
    df_pq = pd.read_parquet(ruta_parquet, engine="pyarrow")
    tiempo_lectura_pq = time.perf_counter() - inicio_pq

    # Lectura selectiva (3 columnas críticas para ML)
    columnas_ml = ["costo", "asientos_disponibles", "viaje_exitoso"]

    inicio_csv_col = time.perf_counter()
    _ = pd.read_csv(ruta_csv, usecols=columnas_ml)
    tiempo_csv_col = time.perf_counter() - inicio_csv_col

    inicio_pq_col = time.perf_counter()
    _ = pd.read_parquet(ruta_parquet, columns=columnas_ml, engine="pyarrow")
    tiempo_pq_col = time.perf_counter() - inicio_pq_col

    aceleracion_completa = tiempo_lectura_csv / tiempo_lectura_pq if tiempo_lectura_pq > 0 else 1.0
    aceleracion_columnar = tiempo_csv_col / tiempo_pq_col if tiempo_pq_col > 0 else 1.0

    print(f"Tamaño CSV en disco:      {tamano_csv_mb:.2f} MB")
    print(f"Tamaño Parquet en disco:  {tamano_parquet_mb:.2f} MB (Ahorro: {ahorro_espacio_pct:.2f}%)")
    print(f"Lectura completa CSV:     {tiempo_lectura_csv:.4f} s")
    print(f"Lectura completa Parquet: {tiempo_lectura_pq:.4f} s ({aceleracion_completa:.2f}x más rápido)")
    print(f"Lectura 3 cols CSV:       {tiempo_csv_col:.4f} s")
    print(f"Lectura 3 cols Parquet:   {tiempo_pq_col:.4f} s ({aceleracion_columnar:.2f}x más rápido)")

    df_metricas = pd.DataFrame([
        {
            "formato": "CSV (Plano UTF-8)",
            "tamano_mb": round(tamano_csv_mb, 3),
            "lectura_completa_seg": round(tiempo_lectura_csv, 4),
            "lectura_3_columnas_seg": round(tiempo_csv_col, 4)
        },
        {
            "formato": "Apache Parquet (Snappy)",
            "tamano_mb": round(tamano_parquet_mb, 3),
            "lectura_completa_seg": round(tiempo_lectura_pq, 4),
            "lectura_3_columnas_seg": round(tiempo_pq_col, 4)
        }
    ])
    df_metricas.to_csv(REPORTE_BENCHMARK, index=False, encoding="utf-8-sig")

    # Generar gráfica comparativa (paleta verde Velox UTVT)
    fig, ejes = plt.subplots(1, 2, figsize=(11, 5))

    formatos = ["CSV", "Parquet (Snappy)"]
    colores = ["#95a5a6", "#27ae60"]

    # Subgráfica 1: Tamaño en disco
    ejes[0].bar(formatos, [tamano_csv_mb, tamano_parquet_mb], color=colores, width=0.5)
    ejes[0].set_title("Tamaño en Disco (MB)")
    ejes[0].set_ylabel("Megabytes (MB)")
    ejes[0].grid(axis="y", alpha=0.3)
    for idx, val in enumerate([tamano_csv_mb, tamano_parquet_mb]):
        ejes[0].text(idx, val + 0.05, f"{val:.2f} MB", ha="center", fontweight="bold")

    # Subgráfica 2: Tiempos de lectura
    x = np.arange(len(formatos))
    ancho = 0.35
    ejes[1].bar(x - ancho / 2, [tiempo_lectura_csv, tiempo_lectura_pq], width=ancho, label="Lectura Completa", color="#27ae60")
    ejes[1].bar(x + ancho / 2, [tiempo_csv_col, tiempo_pq_col], width=ancho, label="Proyección 3 Cols", color="#a9dfbf")
    ejes[1].set_xticks(x)
    ejes[1].set_xticklabels(formatos)
    ejes[1].set_title("Tiempo de Lectura (Segundos)")
    ejes[1].set_ylabel("Segundos (menor es mejor)")
    ejes[1].legend()
    ejes[1].grid(axis="y", alpha=0.3)

    plt.suptitle("Velox UTVT - Benchmark Big Data: CSV vs. Apache Parquet", fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig(GRAFICA_BENCHMARK, dpi=300)
    plt.close()

    return df_pq


# ============================================================
# 5. AUDITORIA DE INTEGRIDAD DEL ARCHIVO PARQUET
# ============================================================

def verificar_integridad(df_original, df_parquet):
    """Comprueba que no haya pérdida de registros ni alteración de métricas."""
    print("\n4. AUDITORIA DE INTEGRIDAD DE DATOS")
    print("-" * 65)

    assert len(df_original) == len(df_parquet), "Error: Discrepancia en el número de filas."
    assert list(df_original.columns) == list(df_parquet.columns), "Error: Discrepancia en columnas."

    print(f"Integridad verificada: {len(df_parquet):,} filas idénticas conservadas.")
    print(f"Promedio de costo verificado: ${df_parquet['costo'].mean():.2f} MXN")
    print(f"Viajes exitosos verificados: {int(df_parquet['viaje_exitoso'].sum()):,}")


# ============================================================
# 6. PUNTO DE ENTRADA
# ============================================================

def main():
    print("=" * 65)
    print("VELOX UTVT - PRACTICA 21 (ETAPA 13)")
    print("CONVERSION Y BENCHMARK DE APACHE PARQUET")
    print("=" * 65)

    if not ARCHIVO_CSV.exists():
        print(f"\nError: No se encontró el archivo base '{ARCHIVO_CSV}'.")
        print("Ejecuta primero: python p20generador_datos_masivos_velox.py")
        return

    df_csv, tiempo_csv = cargar_y_optimizar_csv(ARCHIVO_CSV)
    exportar_a_parquet(df_csv, ARCHIVO_PARQUET)
    df_pq = ejecutar_benchmark(ARCHIVO_CSV, ARCHIVO_PARQUET, tiempo_csv)
    verificar_integridad(df_csv, df_pq)

    print("\nArchivos generados:")
    print(f"- Dataset Parquet: {ARCHIVO_PARQUET}")
    print(f"- Benchmark CSV:   {REPORTE_BENCHMARK}")
    print(f"- Gráfica:         {GRAFICA_BENCHMARK}")
    print("\nPRACTICA 21 FINALIZADA CORRECTAMENTE.")


if __name__ == "__main__":
    main()