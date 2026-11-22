# ============================================================
# PROYECTO: VELOX UTVT
# PRACTICA 22 (ETAPA 14): PROCESAMIENTO DISTRIBUIDO CON PYSPARK
#
# Archivo:
# p22spark_procesamiento_velox.py
#
# Objetivo:
# Procesar de forma distribuida el dataset masivo Parquet de
# Velox UTVT mediante Apache Spark (DataFrames y Spark SQL),
# generando ingeniería de variables, agregaciones por ruta y
# horario universitario, y exportación para Spark MLlib.
#
# Ejecución desde terminal:
# python p22spark_procesamiento_velox.py
# ============================================================

import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pyspark.sql import SparkSession
from pyspark.sql import functions as F


# ============================================================
# 1. CONFIGURACION DE RUTAS
# ============================================================

RUTA_ACTUAL = Path(__file__).resolve().parent
CARPETA_BIGDATA = RUTA_ACTUAL / "resultados_bigdata_velox"
CARPETA_BIGDATA.mkdir(parents=True, exist_ok=True)

ARCHIVO_PARQUET_ENTRADA = CARPETA_BIGDATA / "velox_viajes_masivos_50k.parquet"
ARCHIVO_PARQUET_SALIDA = CARPETA_BIGDATA / "velox_viajes_procesados_spark.parquet"
REPORTE_RUTAS_CSV = CARPETA_BIGDATA / "p22_spark_resumen_rutas.csv"
REPORTE_HORARIOS_CSV = CARPETA_BIGDATA / "p22_spark_resumen_horarios.csv"
GRAFICA_RUTAS = CARPETA_BIGDATA / "p22_spark_ingresos_por_ruta.png"


# ============================================================
# 2. INICIALIZAR SESION DE APACHE SPARK
# ============================================================

def crear_sesion_spark():
    """
    Inicializa una sesión local de Apache Spark optimizada para
    procesamiento analítico en memoria.
    """
    spark = (
        SparkSession.builder
        .appName("VeloxUTVT_BigData_PySpark")
        .master("local[*]")
        .config("spark.sql.shuffle.partitions", "4")
        .config("spark.driver.memory", "2g")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("ERROR")
    return spark


# ============================================================
# 3. INGENIERIA DE VARIABLES DISTRIBUIDA
# ============================================================

def transformar_datos_spark(df_spark):
    """
    Aplica transformaciones vectorizadas en Spark para enriquecer
    el dataset con métricas económicas y operativas de Velox UTVT.
    """
    print("\n1. ESQUEMA ORIGINAL E INGENIERIA DE VARIABLES (SPARK)")
    print("-" * 65)
    df_spark.printSchema()

    df_enriquecido = (
        df_spark
        # Ingreso máximo potencial si se ocupan todos los asientos ofertados
        .withColumn(
            "ingreso_potencial_viaje",
            F.round(F.col("costo") * F.col("asientos_disponibles"), 2)
        )
        # Tarifa por kilómetro recorrido hacia Campus UTVT
        .withColumn(
            "costo_por_km",
            F.round(F.col("costo") / F.col("distancia_km"), 2)
        )
        # Clasificación tarifaria institucional
        .withColumn(
            "segmento_tarifario",
            F.when(F.col("costo") <= 15.0, "Economico")
            .when(F.col("costo") <= 25.0, "Estandar")
            .otherwise("Tarifa_Alta")
        )
        # Etiqueta legible de consolidación
        .withColumn(
            "viabilidad_operativa",
            F.when(F.col("viaje_exitoso") == 1, "Consolidado")
            .otherwise("No_Consolidado")
        )
    )

    print("Muestra de registros enriquecidos en Spark:")
    df_enriquecido.select(
        "id_viaje", "origen", "costo", "asientos_disponibles",
        "ingreso_potencial_viaje", "costo_por_km", "segmento_tarifario", "viabilidad_operativa"
    ).show(5, truncate=False)

    return df_enriquecido


# ============================================================
# 4. AGREGACIONES POR RUTA Y HORARIO (DATAFRAMES + SPARK SQL)
# ============================================================

def analizar_con_spark_sql(spark, df_enriquecido):
    """
    Registra una vista temporal y ejecuta consultas analíticas
    distribuidas por punto de origen y horario universitario.
    """
    print("\n2. ANALISIS DISTRIBUIDO POR PUNTO DE ORIGEN (SPARK SQL)")
    print("-" * 65)

    df_enriquecido.createOrReplaceTempView("viajes_velox")

    consulta_rutas = """
        SELECT
            origen,
            COUNT(*) AS total_viajes,
            ROUND(AVG(distancia_km), 2) AS distancia_promedio_km,
            ROUND(AVG(costo), 2) AS tarifa_promedio_mxn,
            ROUND(AVG(asientos_disponibles), 2) AS asientos_promedio,
            ROUND(SUM(ingreso_potencial_viaje), 2) AS ingreso_potencial_total,
            ROUND(AVG(viaje_exitoso) * 100, 2) AS tasa_consolidacion_pct
        FROM viajes_velox
        GROUP BY origen
        ORDER BY total_viajes DESC
    """
    df_rutas = spark.sql(consulta_rutas)
    df_rutas.show(truncate=False)

    print("\n3. ANALISIS POR HORARIO UNIVERSITARIO UTVT")
    print("-" * 65)

    df_horarios = (
        df_enriquecido
        .groupBy("horario")
        .agg(
            F.count("*").alias("total_viajes"),
            F.round(F.avg("costo"), 2).alias("tarifa_promedio"),
            F.round(F.avg("puntualidad"), 2).alias("puntualidad_promedio"),
            F.round(F.avg("viaje_exitoso") * 100, 2).alias("tasa_exito_pct")
        )
        .orderBy("horario")
    )
    df_horarios.show(truncate=False)

    print("\n4. DISTRIBUCION POR SEGMENTO TARIFARIO")
    print("-" * 65)

    df_segmentos = (
        df_enriquecido
        .groupBy("segmento_tarifario")
        .agg(
            F.count("*").alias("cantidad_viajes"),
            F.round(F.avg("costo"), 2).alias("costo_medio"),
            F.round(F.avg("viaje_exitoso") * 100, 2).alias("tasa_exito_pct")
        )
        .orderBy(F.desc("cantidad_viajes"))
    )
    df_segmentos.show(truncate=False)

    return df_rutas, df_horarios


# ============================================================
# 5. EXPORTACION DE RESULTADOS Y GRAFICA
# ============================================================

def exportar_resultados_spark(df_enriquecido, df_rutas, df_horarios):
    """
    Exporta las tablas agregadas a CSV, guarda el dataset enriquecido
    en Parquet y genera una gráfica de barras en tonos verdes.
    """
    print("\n5. EXPORTACION DE RESULTADOS DISTRIBUIDOS")
    print("-" * 65)

    pdf_rutas = df_rutas.toPandas()
    pdf_horarios = df_horarios.toPandas()

    pdf_rutas.to_csv(REPORTE_RUTAS_CSV, index=False, encoding="utf-8-sig")
    pdf_horarios.to_csv(REPORTE_HORARIOS_CSV, index=False, encoding="utf-8-sig")

    # Guardar dataset enriquecido en Parquet para la Práctica 23 (Spark MLlib)
    df_enriquecido.write.mode("overwrite").parquet(str(ARCHIVO_PARQUET_SALIDA))

    # Gráfica de tasa de consolidación y tarifa promedio por origen
    plt.figure(figsize=(10, 5))
    plt.barh(pdf_rutas["origen"], pdf_rutas["tasa_consolidacion_pct"], color="#27ae60")
    plt.xlabel("Tasa de Consolidación de Viajes (%)")
    plt.ylabel("Punto de Origen (Valle de Toluca)")
    plt.title("Velox UTVT - Tasa de Éxito por Ruta (Procesamiento PySpark)")
    plt.xlim(0, 100)
    plt.grid(axis="x", alpha=0.3)

    for idx, val in enumerate(pdf_rutas["tasa_consolidacion_pct"]):
        plt.text(val + 1, idx, f"{val:.1f}%", va="center", fontweight="bold")

    plt.tight_layout()
    plt.savefig(GRAFICA_RUTAS, dpi=300)
    plt.close()

    print(f"- Resumen de rutas CSV:      {REPORTE_RUTAS_CSV}")
    print(f"- Resumen de horarios CSV:   {REPORTE_HORARIOS_CSV}")
    print(f"- Dataset Parquet Spark:     {ARCHIVO_PARQUET_SALIDA}")
    print(f"- Gráfica generada:          {GRAFICA_RUTAS}")


# ============================================================
# 6. PUNTO DE ENTRADA
# ============================================================

def main():
    print("=" * 65)
    print("VELOX UTVT - PRACTICA 22 (ETAPA 14)")
    print("PROCESAMIENTO DISTRIBUIDO DE BIG DATA CON APACHE SPARK")
    print("=" * 65)

    if not ARCHIVO_PARQUET_ENTRADA.exists():
        print(f"\nError: No se encontró '{ARCHIVO_PARQUET_ENTRADA}'.")
        print("Ejecuta primero: python p21conversion_parquet_velox.py")
        return

    inicio = time.perf_counter()
    spark = crear_sesion_spark()

    try:
        df_spark = spark.read.parquet(str(ARCHIVO_PARQUET_ENTRADA))
        print(f"Total de registros cargados en Spark: {df_spark.count():,}")

        df_enriquecido = transformar_datos_spark(df_spark)
        df_rutas, df_horarios = analizar_con_spark_sql(spark, df_enriquecido)
        exportar_resultados_spark(df_enriquecido, df_rutas, df_horarios)

        duracion = time.perf_counter() - inicio
        print(f"\nTiempo total de procesamiento distribuido: {duracion:.2f} segundos")
        print("PRACTICA 22 FINALIZADA CORRECTAMENTE.")
    finally:
        spark.stop()


if __name__ == "__main__":
    main()