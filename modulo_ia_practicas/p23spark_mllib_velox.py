# ============================================================
# PROYECTO: VELOX UTVT
# PRACTICA 23 (ETAPA 15): MACHINE LEARNING DISTRIBUIDO CON SPARK MLLIB
#
# Archivo:
# p23spark_mllib_velox.py
#
# Objetivo:
# Construir un Pipeline distribuido de Machine Learning con Spark
# MLlib sobre el dataset Parquet procesado de Velox UTVT para
# predecir la consolidación de viajes (viaje_exitoso), comparando
# Logistic Regression, Decision Tree y Random Forest a gran escala.
#
# Ejecución desde terminal:
# python p23spark_mllib_velox.py
# ============================================================

import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.ml import Pipeline
from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorAssembler, StandardScaler
from pyspark.ml.classification import (
    LogisticRegression,
    DecisionTreeClassifier,
    RandomForestClassifier
)
from pyspark.ml.evaluation import (
    BinaryClassificationEvaluator,
    MulticlassClassificationEvaluator
)


# ============================================================
# 1. CONFIGURACION DE RUTAS Y PARAMETROS
# ============================================================

SEMILLA = 42
RUTA_ACTUAL = Path(__file__).resolve().parent
CARPETA_BIGDATA = RUTA_ACTUAL / "resultados_bigdata_velox"
CARPETA_BIGDATA.mkdir(parents=True, exist_ok=True)

ARCHIVO_PARQUET_SPARK = CARPETA_BIGDATA / "velox_viajes_procesados_spark.parquet"
ARCHIVO_PARQUET_RESPALDO = CARPETA_BIGDATA / "velox_viajes_masivos_50k.parquet"
REPORTE_MLLIB_CSV = CARPETA_BIGDATA / "p23_mllib_comparacion_modelos.csv"
GRAFICA_MLLIB = CARPETA_BIGDATA / "p23_mllib_metricas.png"
RUTA_MODELO_GANADOR = CARPETA_BIGDATA / "modelo_mllib_ganador_velox"


# ============================================================
# 2. INICIALIZAR SESION DE APACHE SPARK
# ============================================================

def crear_sesion_spark():
    """Inicializa una sesión local de Spark optimizada para MLlib."""
    spark = (
        SparkSession.builder
        .appName("VeloxUTVT_SparkMLlib")
        .master("local[*]")
        .config("spark.sql.shuffle.partitions", "4")
        .config("spark.driver.memory", "2g")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("ERROR")
    return spark


# ============================================================
# 3. CONSTRUCCION DEL PIPELINE DE PREPROCESAMIENTO
# ============================================================

def construir_etapas_preprocesamiento():
    """
    Define las transformaciones de características en Spark MLlib:
    - Codificación de variable categórica 'origen'
    - Ensamblado vectorial de variables numéricas y categóricas
    - Estandarización con StandardScaler
    """
    indexador_origen = StringIndexer(
        inputCol="origen",
        outputCol="origen_idx",
        handleInvalid="keep"
    )
    encoder_origen = OneHotEncoder(
        inputCols=["origen_idx"],
        outputCols=["origen_vec"]
    )

    columnas_numericas = [
        "distancia_km",
        "costo",
        "asientos_disponibles",
        "puntualidad",
        "calificacion_conductor",
        "viajes_previos"
    ]

    ensamblador = VectorAssembler(
        inputCols=columnas_numericas + ["origen_vec"],
        outputCol="features_raw"
    )

    escalador = StandardScaler(
        inputCol="features_raw",
        outputCol="features",
        withStd=True,
        withMean=True
    )

    return [indexador_origen, encoder_origen, ensamblador, escalador]


# ============================================================
# 4. ENTRENAMIENTO Y EVALUACION DISTRIBUIDA DE MODELOS
# ============================================================

def entrenar_y_evaluar_modelos(df_train, df_test):
    """
    Entrena y evalúa 3 algoritmos distribuidos de Spark MLlib:
    - Regresión Logística
    - Árbol de Decisión
    - Random Forest
    """
    print("\n2. ENTRENAMIENTO Y EVALUACION EN SPARK MLLIB")
    print("-" * 65)

    etapas_base = construir_etapas_preprocesamiento()

    clasificadores = {
        "Logistic Regression (MLlib)": LogisticRegression(
            featuresCol="features",
            labelCol="label",
            maxIter=30,
            regParam=0.01
        ),
        "Decision Tree (MLlib)": DecisionTreeClassifier(
            featuresCol="features",
            labelCol="label",
            maxDepth=6,
            seed=SEMILLA
        ),
        "Random Forest (MLlib)": RandomForestClassifier(
            featuresCol="features",
            labelCol="label",
            numTrees=80,
            maxDepth=7,
            seed=SEMILLA
        )
    }

    eval_auc = BinaryClassificationEvaluator(
        labelCol="label",
        rawPredictionCol="rawPrediction",
        metricName="areaUnderROC"
    )
    eval_acc = MulticlassClassificationEvaluator(
        labelCol="label",
        predictionCol="prediction",
        metricName="accuracy"
    )
    eval_f1 = MulticlassClassificationEvaluator(
        labelCol="label",
        predictionCol="prediction",
        metricName="f1"
    )
    eval_prec = MulticlassClassificationEvaluator(
        labelCol="label",
        predictionCol="prediction",
        metricName="weightedPrecision"
    )
    eval_rec = MulticlassClassificationEvaluator(
        labelCol="label",
        predictionCol="prediction",
        metricName="weightedRecall"
    )

    resultados = []
    pipelines_entrenados = {}

    for nombre, clasificador in clasificadores.items():
        print(f"\nEntrenando modelo distribuido: {nombre}...")
        inicio_m = time.perf_counter()

        pipeline = Pipeline(stages=etapas_base + [clasificador])
        modelo_ajustado = pipeline.fit(df_train)
        predicciones = modelo_ajustado.transform(df_test)

        tiempo_m = time.perf_counter() - inicio_m

        auc = float(eval_auc.evaluate(predicciones))
        acc = float(eval_acc.evaluate(predicciones))
        f1 = float(eval_f1.evaluate(predicciones))
        prec = float(eval_prec.evaluate(predicciones))
        rec = float(eval_rec.evaluate(predicciones))

        print(f"  AUC-ROC:   {auc:.4f}")
        print(f"  Accuracy:  {acc * 100:.2f}%")
        print(f"  Precision: {prec * 100:.2f}%")
        print(f"  Recall:    {rec * 100:.2f}%")
        print(f"  F1-Score:  {f1 * 100:.2f}%")
        print(f"  Tiempo:    {tiempo_m:.2f} s")

        resultados.append({
            "algoritmo": nombre,
            "auc_roc": round(auc, 4),
            "accuracy": round(acc, 4),
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "f1_score": round(f1, 4),
            "tiempo_seg": round(tiempo_m, 2)
        })
        pipelines_entrenados[nombre] = (modelo_ajustado, predicciones)

    df_resultados = pd.DataFrame(resultados)
    return df_resultados, pipelines_entrenados


# ============================================================
# 5. EXPORTACION DE METRICAS, GRAFICA Y MODELO GANADOR
# ============================================================

def exportar_resultados_mllib(df_resultados, pipelines_entrenados):
    """
    Exporta la tabla comparativa a CSV, genera la gráfica de
    métricas y guarda el pipeline ganador en disco.
    """
    print("\n3. EXPORTACION DE RESULTADOS Y MODELO GANADOR")
    print("-" * 65)

    df_resultados.to_csv(REPORTE_MLLIB_CSV, index=False, encoding="utf-8-sig")

    # Seleccionar el mejor modelo por F1-score
    idx_mejor = df_resultados["f1_score"].idxmax()
    nombre_mejor = df_resultados.loc[idx_mejor, "algoritmo"]
    mejor_f1 = df_resultados.loc[idx_mejor, "f1_score"]
    mejor_auc = df_resultados.loc[idx_mejor, "auc_roc"]

    modelo_ganador, pred_ganador = pipelines_entrenados[nombre_mejor]
    modelo_ganador.write().overwrite().save(str(RUTA_MODELO_GANADOR))

    print(f"Mejor algoritmo distribuido: {nombre_mejor}")
    print(f"F1-Score: {mejor_f1 * 100:.2f}% | AUC-ROC: {mejor_auc:.4f}")
    print(f"Pipeline guardado en: {RUTA_MODELO_GANADOR}")

    print("\nMuestra de predicciones del modelo ganador:")
    pred_ganador.select(
        "id_viaje", "origen", "costo", "asientos_disponibles",
        "label", "prediction", "probability"
    ).show(5, truncate=False)

    # Gráfica comparativa (tonos verdes Velox UTVT)
    x = np.arange(len(df_resultados))
    ancho = 0.25

    plt.figure(figsize=(10, 6))
    plt.bar(x - ancho, df_resultados["accuracy"], width=ancho, label="Accuracy", color="#27ae60")
    plt.bar(x, df_resultados["f1_score"], width=ancho, label="F1-Score", color="#52be80")
    plt.bar(x + ancho, df_resultados["auc_roc"], width=ancho, label="AUC-ROC", color="#a9dfbf")

    plt.xticks(x, df_resultados["algoritmo"])
    plt.ylim(0, 1.08)
    plt.ylabel("Puntaje (0 a 1)")
    plt.title("Velox UTVT - Comparación de Modelos Distribuidos (Spark MLlib)")
    plt.legend()
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(GRAFICA_MLLIB, dpi=300)
    plt.close()

    print(f"- Tabla comparativa CSV: {REPORTE_MLLIB_CSV}")
    print(f"- Gráfica generada:      {GRAFICA_MLLIB}")


# ============================================================
# 6. PUNTO DE ENTRADA
# ============================================================

def main():
    print("=" * 65)
    print("VELOX UTVT - PRACTICA 23 (ETAPA 15)")
    print("MACHINE LEARNING DISTRIBUIDO CON APACHE SPARK MLLIB")
    print("=" * 65)

    ruta_datos = ARCHIVO_PARQUET_SPARK if ARCHIVO_PARQUET_SPARK.exists() else ARCHIVO_PARQUET_RESPALDO
    if not ruta_datos.exists():
        print(f"\nError: No se encontró dataset Parquet en '{CARPETA_BIGDATA}'.")
        print("Ejecuta primero: python p21conversion_parquet_velox.py")
        return

    inicio = time.perf_counter()
    spark = crear_sesion_spark()

    try:
        print("\n1. CARGA DE DATOS EN SPARK MLLIB")
        print("-" * 65)
        df_spark = spark.read.parquet(str(ruta_datos))
        df_ml = df_spark.withColumn("label", F.col("viaje_exitoso").cast("double"))

        df_train, df_test = df_ml.randomSplit([0.80, 0.20], seed=SEMILLA)
        print(f"Fuente Parquet: {ruta_datos.name}")
        print(f"Registros entrenamiento (80%): {df_train.count():,}")
        print(f"Registros prueba (20%):        {df_test.count():,}")

        df_resultados, pipelines_entrenados = entrenar_y_evaluar_modelos(df_train, df_test)
        exportar_resultados_mllib(df_resultados, pipelines_entrenados)

        duracion = time.perf_counter() - inicio
        print(f"\nTiempo total de ejecución Spark MLlib: {duracion:.2f} segundos")
        print("PRACTICA 23 FINALIZADA CORRECTAMENTE.")
    finally:
        spark.stop()


if __name__ == "__main__":
    main()