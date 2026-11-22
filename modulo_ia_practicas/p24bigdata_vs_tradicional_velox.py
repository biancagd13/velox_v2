# ============================================================
# PROYECTO: VELOX UTVT
# PRACTICA 24 (ETAPA 16): BIG DATA VS. PROCESAMIENTO TRADICIONAL
#
# Archivo:
# p24bigdata_vs_tradicional_velox.py
#
# Objetivo:
# Comparar empíricamente el rendimiento en almacenamiento,
# tiempo de lectura, agregación analítica y entrenamiento de
# modelos predictivos entre la arquitectura tradicional
# (CSV + Pandas + Scikit-Learn) y el ecosistema Big Data
# (Parquet + Apache Spark + MLlib) sobre Velox UTVT.
#
# Ejecución desde terminal:
# python p24bigdata_vs_tradicional_velox.py
# ============================================================

import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier as SklearnRF
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline as SklearnPipeline
from sklearn.preprocessing import OneHotEncoder as SkOneHot, StandardScaler as SkScaler

try:
    from pyspark.sql import SparkSession
    from pyspark.sql import functions as F
    from pyspark.ml import Pipeline as SparkPipeline
    from pyspark.ml.feature import (
        StringIndexer,
        OneHotEncoder as SparkOneHot,
        VectorAssembler,
        StandardScaler as SparkScaler
    )
    from pyspark.ml.classification import RandomForestClassifier as SparkRF
    from pyspark.ml.evaluation import MulticlassClassificationEvaluator
    PYSPARK_DISPONIBLE = True
except ImportError:
    PYSPARK_DISPONIBLE = False


# ============================================================
# 1. CONFIGURACION DE RUTAS Y PARAMETROS
# ============================================================

SEMILLA = 42
RUTA_ACTUAL = Path(__file__).resolve().parent
CARPETA_BIGDATA = RUTA_ACTUAL / "resultados_bigdata_velox"
CARPETA_BIGDATA.mkdir(parents=True, exist_ok=True)

ARCHIVO_CSV = CARPETA_BIGDATA / "velox_viajes_masivos_50k.csv"
ARCHIVO_PARQUET = CARPETA_BIGDATA / "velox_viajes_masivos_50k.parquet"
REPORTE_COMPARATIVA = CARPETA_BIGDATA / "p24_comparativa_arquitecturas.csv"
GRAFICA_COMPARATIVA = CARPETA_BIGDATA / "p24_bigdata_vs_tradicional.png"


# ============================================================
# 2. PIPELINE TRADICIONAL: CSV + PANDAS + SCIKIT-LEARN
# ============================================================

def ejecutar_flujo_tradicional(ruta_csv):
    """
    Mide tiempos de lectura CSV, agregación con Pandas y
    entrenamiento de Random Forest con Scikit-Learn.
    """
    print("\n1. ARQUITECTURA TRADICIONAL (CSV + PANDAS + SCIKIT-LEARN)")
    print("-" * 65)

    # Fase A: Lectura desde CSV plano
    t0 = time.perf_counter()
    df = pd.read_csv(ruta_csv, encoding="utf-8")
    t_lectura = time.perf_counter() - t0

    # Fase B: Agregación analítica por punto de origen
    t0 = time.perf_counter()
    _ = (
        df.groupby("origen", as_index=False)
        .agg(
            total_viajes=("id_viaje", "count"),
            tarifa_promedio=("costo", "mean"),
            tasa_exito=("viaje_exitoso", "mean")
        )
    )
    t_agregacion = time.perf_counter() - t0

    # Fase C: Preprocesamiento y entrenamiento Scikit-Learn
    t0 = time.perf_counter()
    columnas_num = [
        "distancia_km", "costo", "asientos_disponibles",
        "puntualidad", "calificacion_conductor", "viajes_previos"
    ]
    columnas_cat = ["origen"]

    X = df[columnas_num + columnas_cat]
    y = df["viaje_exitoso"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=SEMILLA, stratify=y
    )

    preprocesador = ColumnTransformer(
        transformers=[
            ("num", SkScaler(), columnas_num),
            ("cat", SkOneHot(handle_unknown="ignore"), columnas_cat)
        ]
    )

    modelo_sk = SklearnPipeline([
        ("prep", preprocesador),
        ("rf", SklearnRF(n_estimators=60, max_depth=7, random_state=SEMILLA, n_jobs=-1))
    ])

    modelo_sk.fit(X_train, y_train)
    preds = modelo_sk.predict(X_test)
    t_entrenamiento = time.perf_counter() - t0

    acc = float(accuracy_score(y_test, preds))
    f1 = float(f1_score(y_test, preds))
    tamano_mb = ruta_csv.stat().st_size / (1024 * 1024)
    total_seg = t_lectura + t_agregacion + t_entrenamiento

    print(f"  Almacenamiento CSV:       {tamano_mb:.2f} MB")
    print(f"  Tiempo lectura:           {t_lectura:.4f} s")
    print(f"  Tiempo agregación:        {t_agregacion:.4f} s")
    print(f"  Tiempo entrenamiento ML:  {t_entrenamiento:.4f} s")
    print(f"  Tiempo total pipeline:    {total_seg:.4f} s")
    print(f"  Accuracy: {acc * 100:.2f}% | F1-Score: {f1 * 100:.2f}%")

    return {
        "arquitectura": "Tradicional (CSV + Pandas + Sklearn)",
        "tamano_disco_mb": round(tamano_mb, 3),
        "tiempo_lectura_seg": round(t_lectura, 4),
        "tiempo_agregacion_seg": round(t_agregacion, 4),
        "tiempo_ml_seg": round(t_entrenamiento, 4),
        "tiempo_total_seg": round(total_seg, 4),
        "accuracy": round(acc, 4),
        "f1_score": round(f1, 4)
    }


# ============================================================
# 3. PIPELINE BIG DATA: PARQUET + APACHE SPARK + MLLIB
# ============================================================

def ejecutar_flujo_bigdata(ruta_parquet):
    """
    Mide tiempos de lectura columnar Parquet, agregación distribuida
    en Apache Spark y entrenamiento con Spark MLlib.
    """
    print("\n2. ARQUITECTURA BIG DATA (PARQUET + APACHE SPARK + MLLIB)")
    print("-" * 65)

    if not PYSPARK_DISPONIBLE:
        print("Aviso: PySpark no está instalado aún. Usando lectura PyArrow como respaldo.")
        t0 = time.perf_counter()
        df_pq = pd.read_parquet(ruta_parquet, engine="pyarrow")
        t_lectura = time.perf_counter() - t0

        t0 = time.perf_counter()
        _ = df_pq.groupby("origen")["costo"].mean()
        t_agregacion = time.perf_counter() - t0

        tamano_mb = ruta_parquet.stat().st_size / (1024 * 1024)
        return {
            "arquitectura": "Columnar (Parquet + PyArrow)",
            "tamano_disco_mb": round(tamano_mb, 3),
            "tiempo_lectura_seg": round(t_lectura, 4),
            "tiempo_agregacion_seg": round(t_agregacion, 4),
            "tiempo_ml_seg": 0.0,
            "tiempo_total_seg": round(t_lectura + t_agregacion, 4),
            "accuracy": 0.0,
            "f1_score": 0.0
        }

    spark = (
        SparkSession.builder
        .appName("VeloxUTVT_Benchmark_P24")
        .master("local[*]")
        .config("spark.sql.shuffle.partitions", "4")
        .config("spark.driver.memory", "2g")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("ERROR")

    try:
        # Fase A: Lectura columnar Parquet
        t0 = time.perf_counter()
        df_spark = spark.read.parquet(str(ruta_parquet))
        _ = df_spark.count()
        t_lectura = time.perf_counter() - t0

        # Fase B: Agregación distribuida por punto de origen
        t0 = time.perf_counter()
        _ = (
            df_spark.groupBy("origen")
            .agg(
                F.count("*").alias("total_viajes"),
                F.avg("costo").alias("tarifa_promedio"),
                F.avg("viaje_exitoso").alias("tasa_exito")
            )
            .collect()
        )
        t_agregacion = time.perf_counter() - t0

        # Fase C: Pipeline distribuido Spark MLlib
        t0 = time.perf_counter()
        df_ml = df_spark.withColumn("label", F.col("viaje_exitoso").cast("double"))
        df_train, df_test = df_ml.randomSplit([0.80, 0.20], seed=SEMILLA)

        idx = StringIndexer(inputCol="origen", outputCol="origen_idx", handleInvalid="keep")
        ohe = SparkOneHot(inputCols=["origen_idx"], outputCols=["origen_vec"])
        asm = VectorAssembler(
            inputCols=[
                "distancia_km", "costo", "asientos_disponibles",
                "puntualidad", "calificacion_conductor", "viajes_previos", "origen_vec"
            ],
            outputCol="features_raw"
        )
        scl = SparkScaler(inputCol="features_raw", outputCol="features", withStd=True, withMean=True)
        rf_spark = SparkRF(featuresCol="features", labelCol="label", numTrees=60, maxDepth=7, seed=SEMILLA)

        pipeline_spark = SparkPipeline(stages=[idx, ohe, asm, scl, rf_spark])
        modelo_spark = pipeline_spark.fit(df_train)
        preds_spark = modelo_spark.transform(df_test)

        eval_acc = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="accuracy")
        eval_f1 = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="f1")
        acc = float(eval_acc.evaluate(preds_spark))
        f1 = float(eval_f1.evaluate(preds_spark))
        t_entrenamiento = time.perf_counter() - t0

        tamano_mb = ruta_parquet.stat().st_size / (1024 * 1024)
        total_seg = t_lectura + t_agregacion + t_entrenamiento

        print(f"  Almacenamiento Parquet:   {tamano_mb:.2f} MB")
        print(f"  Tiempo lectura:           {t_lectura:.4f} s")
        print(f"  Tiempo agregación:        {t_agregacion:.4f} s")
        print(f"  Tiempo entrenamiento ML:  {t_entrenamiento:.4f} s")
        print(f"  Tiempo total pipeline:    {total_seg:.4f} s")
        print(f"  Accuracy: {acc * 100:.2f}% | F1-Score: {f1 * 100:.2f}%")

        return {
            "arquitectura": "Big Data (Parquet + Spark + MLlib)",
            "tamano_disco_mb": round(tamano_mb, 3),
            "tiempo_lectura_seg": round(t_lectura, 4),
            "tiempo_agregacion_seg": round(t_agregacion, 4),
            "tiempo_ml_seg": round(t_entrenamiento, 4),
            "tiempo_total_seg": round(total_seg, 4),
            "accuracy": round(acc, 4),
            "f1_score": round(f1, 4)
        }
    finally:
        spark.stop()


# ============================================================
# 4. EXPORTACION Y GRAFICA COMPARATIVA
# ============================================================

def generar_reporte_y_grafica(res_trad, res_bd):
    """Exporta tabla comparativa CSV y genera gráfica en tonos verdes."""
    print("\n3. RESUMEN COMPARATIVO Y EXPORTACION")
    print("-" * 65)

    df_comp = pd.DataFrame([res_trad, res_bd])
    print(df_comp.to_string(index=False))

    df_comp.to_csv(REPORTE_COMPARATIVA, index=False, encoding="utf-8-sig")

    fig, ejes = plt.subplots(1, 2, figsize=(12, 5))

    etiquetas = ["Tradicional\n(CSV + Pandas)", "Big Data\n(Parquet + Spark)"]
    colores = ["#95a5a6", "#27ae60"]

    # Subgráfica 1: Desglose de tiempos por etapa
    fases = ["Lectura", "Agregación", "Entrenamiento ML"]
    tiempos_trad = [
        res_trad["tiempo_lectura_seg"],
        res_trad["tiempo_agregacion_seg"],
        res_trad["tiempo_ml_seg"]
    ]
    tiempos_bd = [
        res_bd["tiempo_lectura_seg"],
        res_bd["tiempo_agregacion_seg"],
        res_bd["tiempo_ml_seg"]
    ]

    x = np.arange(len(fases))
    ancho = 0.35
    ejes[0].bar(x - ancho / 2, tiempos_trad, width=ancho, label="Tradicional (Pandas/Sklearn)", color="#7f8c8d")
    ejes[0].bar(x + ancho / 2, tiempos_bd, width=ancho, label="Big Data (Spark/MLlib)", color="#27ae60")
    ejes[0].set_xticks(x)
    ejes[0].set_xticklabels(fases)
    ejes[0].set_ylabel("Tiempo (Segundos)")
    ejes[0].set_title("Tiempos de Ejecución por Etapa")
    ejes[0].legend()
    ejes[0].grid(axis="y", alpha=0.3)

    # Subgráfica 2: Almacenamiento en disco (MB)
    tamanos = [res_trad["tamano_disco_mb"], res_bd["tamano_disco_mb"]]
    ejes[1].bar(etiquetas, tamanos, color=colores, width=0.45)
    ejes[1].set_ylabel("Tamaño en Disco (MB)")
    ejes[1].set_title("Huella de Almacenamiento (50,000 Viajes)")
    ejes[1].grid(axis="y", alpha=0.3)
    for idx, val in enumerate(tamanos):
        ejes[1].text(idx, val + 0.05, f"{val:.2f} MB", ha="center", fontweight="bold")

    plt.suptitle("Velox UTVT - Comparativa Big Data vs. Procesamiento Tradicional", fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig(GRAFICA_COMPARATIVA, dpi=300)
    plt.close()

    print(f"\n- Reporte CSV guardado: {REPORTE_COMPARATIVA}")
    print(f"- Gráfica guardada:     {GRAFICA_COMPARATIVA}")


# ============================================================
# 5. PUNTO DE ENTRADA
# ============================================================

def main():
    print("=" * 65)
    print("VELOX UTVT - PRACTICA 24 (ETAPA 16)")
    print("BENCHMARK: BIG DATA VS. PROCESAMIENTO TRADICIONAL")
    print("=" * 65)

    if not ARCHIVO_CSV.exists() or not ARCHIVO_PARQUET.exists():
        print(f"\nError: Faltan los datasets base en '{CARPETA_BIGDATA}'.")
        print("Ejecuta primero las prácticas 20 y 21.")
        return

    res_trad = ejecutar_flujo_tradicional(ARCHIVO_CSV)
    res_bd = ejecutar_flujo_bigdata(ARCHIVO_PARQUET)
    generar_reporte_y_grafica(res_trad, res_bd)

    print("\nPRACTICA 24 FINALIZADA CORRECTAMENTE.")


if __name__ == "__main__":
    main()