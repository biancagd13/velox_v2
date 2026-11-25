# ================================================================
# PROYECTO: VELOX UTVT
# PRACTICA 8: ANALISIS AVANZADO Y OPTIMIZACION DE SEGMENTOS DE VIAJES
# ARCHIVO: practica08.py
# ================================================================

import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from database import db

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import silhouette_score, silhouette_samples, adjusted_rand_score

NUM_CLUSTERS = 3
SEMILLA = 42
NUM_INICIALIZACIONES = 20
PERCENTIL_ATIPICOS = 95
CARPETA_RESULTADOS = "resultados_p08_velox"

os.makedirs(CARPETA_RESULTADOS, exist_ok=True)

VARIABLES = [
    "costo",
    "asientos_disponibles"
]

def cargar_datos():
    print("\n" + "=" * 60)
    print("CARGA DE DATOS (MONGODB / VELOX UTVT)")
    print("=" * 60)

    lista_viajes = []
    try:
        cursor = db.viaje.find({}, {"_id": 0, "costo": 1, "asientos_disponibles": 1, "estado": 1})
        for documento in cursor:
            costo = float(documento.get("costo", 0))
            asientos = int(documento.get("asientos_disponibles", 0))
            lista_viajes.append({
                "costo": costo,
                "asientos_disponibles": asientos
            })
    except Exception as error:
        print(f"Error al conectar con MongoDB: {error}")

    df = pd.DataFrame(lista_viajes) if lista_viajes else pd.DataFrame(columns=VARIABLES)

    if df.empty or len(df) < 50:
        print("\nAviso: Pocos registros en MongoDB, generando dataset sintético complementario...")
        rng = np.random.default_rng(SEMILLA)
        n_sinteticos = 500 - len(df) if len(df) < 500 else 0
        if n_sinteticos > 0:
            sintetico_costo = rng.uniform(20, 300, n_sinteticos)
            sintetico_asientos = rng.integers(1, 6, n_sinteticos)
            df_ext = pd.DataFrame({
                "costo": sintetico_costo,
                "asientos_disponibles": sintetico_asientos
            })
            df = pd.concat([df, df_ext], ignore_index=True)

    if "id_viaje" not in df.columns:
        df.insert(
            0,
            "id_viaje",
            [f"VIAJE-{i:04d}" for i in range(1, len(df) + 1)]
        )

    print("Total de viajes cargados:", len(df))
    return df

def preparar_datos(df):
    print("\n" + "=" * 60)
    print("PREPARACION DE DATOS")
    print("=" * 60)

    print("\nValores faltantes por variable:")
    print(df[VARIABLES].isnull().sum())

    for columna in VARIABLES:
        df[columna] = pd.to_numeric(df[columna], errors="coerce")

    df[VARIABLES] = df[VARIABLES].fillna(df[VARIABLES].median())

    df["costo"] = df["costo"].clip(10, 500)
    df["asientos_disponibles"] = df["asientos_disponibles"].clip(1, 6)

    X = df[VARIABLES].to_numpy()
    return df, X

def comparar_inicializaciones(X):
    print("\n" + "=" * 60)
    print("COMPARACION DE INICIALIZACIONES")
    print("=" * 60)

    escalador = StandardScaler()
    X_escalado = escalador.fit_transform(X)

    resultados = []
    modelos = {}

    for inicializacion in ["k-means++", "random"]:
        modelo = KMeans(
            n_clusters=NUM_CLUSTERS,
            init=inicializacion,
            n_init=NUM_INICIALIZACIONES,
            random_state=SEMILLA
        )
        etiquetas = modelo.fit_predict(X_escalado)
        silhouette = silhouette_score(X_escalado, etiquetas)

        resultados.append({
            "inicializacion": inicializacion,
            "inercia": modelo.inertia_,
            "silhouette": silhouette
        })
        modelos[inicializacion] = modelo

        print(f"\nInicializacion: {inicializacion}")
        print("Inercia:", round(modelo.inertia_, 4))
        print("Silhouette:", round(silhouette, 4))

    df_resultados = pd.DataFrame(resultados)
    df_resultados.to_csv(f"{CARPETA_RESULTADOS}/p08_inicializaciones.csv", index=False)

    plt.figure(figsize=(8, 5))
    plt.bar(df_resultados["inicializacion"], df_resultados["silhouette"], color="#27ae60")
    plt.title("Velox UTVT - Comparacion de inicializaciones K-Means")
    plt.xlabel("Metodo de inicializacion")
    plt.ylabel("Silhouette Score")
    plt.tight_layout()
    plt.savefig(f"{CARPETA_RESULTADOS}/p08_inicializaciones.png", dpi=300)
    plt.close()

    return df_resultados, modelos, X_escalado, escalador

def comparar_escaladores(X):
    print("\n" + "=" * 60)
    print("COMPARACION DE ESCALADORES")
    print("=" * 60)

    escaladores = {
        "StandardScaler": StandardScaler(),
        "MinMaxScaler": MinMaxScaler()
    }

    resultados = []
    modelos = {}

    for nombre, escalador in escaladores.items():
        X_transformado = escalador.fit_transform(X)
        modelo = KMeans(
            n_clusters=NUM_CLUSTERS,
            init="k-means++",
            n_init=NUM_INICIALIZACIONES,
            random_state=SEMILLA
        )
        etiquetas = modelo.fit_predict(X_transformado)
        silhouette = silhouette_score(X_transformado, etiquetas)

        resultados.append({
            "escalador": nombre,
            "inercia": modelo.inertia_,
            "silhouette": silhouette
        })
        modelos[nombre] = {
            "modelo": modelo,
            "escalador": escalador,
            "datos_escalados": X_transformado,
            "etiquetas": etiquetas
        }

        print(f"\nEscalador: {nombre}")
        print("Inercia:", round(modelo.inertia_, 4))
        print("Silhouette:", round(silhouette, 4))

    df_resultados = pd.DataFrame(resultados)
    df_resultados.to_csv(f"{CARPETA_RESULTADOS}/p08_comparacion_escaladores.csv", index=False)

    plt.figure(figsize=(8, 5))
    plt.bar(df_resultados["escalador"], df_resultados["silhouette"], color="#27ae60")
    plt.title("Velox UTVT - Comparacion de escaladores")
    plt.xlabel("Escalador")
    plt.ylabel("Silhouette Score")
    plt.tight_layout()
    plt.savefig(f"{CARPETA_RESULTADOS}/p08_escaladores.png", dpi=300)
    plt.close()

    return df_resultados, modelos

def analizar_estabilidad(X_escalado):
    print("\n" + "=" * 60)
    print("ANALISIS DE ESTABILIDAD (ARI)")
    print("=" * 60)

    semillas = [10, 20, 30, 40, 50, 42]
    particiones = {}

    for semilla in semillas:
        modelo = KMeans(
            n_clusters=NUM_CLUSTERS,
            init="k-means++",
            n_init=NUM_INICIALIZACIONES,
            random_state=semilla
        )
        etiquetas = modelo.fit_predict(X_escalado)
        particiones[semilla] = etiquetas

    resultados = []
    for i in range(len(semillas)):
        for j in range(i + 1, len(semillas)):
            semilla_a = semillas[i]
            semilla_b = semillas[j]
            ari = adjusted_rand_score(particiones[semilla_a], particiones[semilla_b])
            resultados.append({
                "semilla_a": semilla_a,
                "semilla_b": semilla_b,
                "ARI": ari
            })

    df_ari = pd.DataFrame(resultados)
    print("\nComparaciones entre ejecuciones:")
    print(df_ari.round(4).to_string(index=False))
    print("\nARI promedio:", round(df_ari["ARI"].mean(), 4))

    df_ari.to_csv(f"{CARPETA_RESULTADOS}/p08_estabilidad_ari.csv", index=False)

    matriz = np.eye(len(semillas))
    for _, fila in df_ari.iterrows():
        i = semillas.index(int(fila["semilla_a"]))
        j = semillas.index(int(fila["semilla_b"]))
        matriz[i, j] = fila["ARI"]
        matriz[j, i] = fila["ARI"]

    plt.figure(figsize=(8, 6))
    imagen = plt.imshow(matriz, vmin=0, vmax=1, cmap="Greens")
    plt.colorbar(imagen, label="Adjusted Rand Index")
    plt.xticks(range(len(semillas)), semillas)
    plt.yticks(range(len(semillas)), semillas)
    plt.title("Velox UTVT - Estabilidad de K-Means (ARI)")
    plt.xlabel("Semilla")
    plt.ylabel("Semilla")

    for i in range(len(semillas)):
        for j in range(len(semillas)):
            plt.text(j, i, f"{matriz[i, j]:.2f}", ha="center", va="center", color="black" if matriz[i, j] > 0.5 else "white")

    plt.tight_layout()
    plt.savefig(f"{CARPETA_RESULTADOS}/p08_matriz_estabilidad.png", dpi=300)
    plt.close()

    return df_ari, particiones

def entrenar_modelo_final(X):
    escalador = StandardScaler()
    X_escalado = escalador.fit_transform(X)

    modelo = KMeans(
        n_clusters=NUM_CLUSTERS,
        init="k-means++",
        n_init=NUM_INICIALIZACIONES,
        random_state=SEMILLA
    )
    etiquetas = modelo.fit_predict(X_escalado)
    silhouette_individual = silhouette_samples(X_escalado, etiquetas)

    return modelo, escalador, X_escalado, etiquetas, silhouette_individual

def analizar_centroides(df, modelo, escalador, etiquetas):
    print("\n" + "=" * 60)
    print("ANALISIS DE CENTROIDES")
    print("=" * 60)

    centroides_originales = escalador.inverse_transform(modelo.cluster_centers_)
    df_centroides = pd.DataFrame(centroides_originales, columns=VARIABLES)
    df_centroides.insert(0, "cluster", range(NUM_CLUSTERS))

    print("\nCentroides en unidades originales:")
    print(df_centroides.round(2).to_string(index=False))

    df_perfiles = df[VARIABLES].copy()
    df_perfiles["cluster"] = etiquetas
    perfiles = df_perfiles.groupby("cluster")[VARIABLES].mean().reset_index()

    print("\nPerfiles promedio de los viajes:")
    print(perfiles.round(2).to_string(index=False))

    df_centroides.to_csv(f"{CARPETA_RESULTADOS}/p08_centroides.csv", index=False)
    perfiles.to_csv(f"{CARPETA_RESULTADOS}/p08_perfiles.csv", index=False)

    return df_centroides, perfiles

def detectar_atipicos(df, X_escalado, modelo, etiquetas, silhouette_individual):
    print("\n" + "=" * 60)
    print("DETECCION DE POSIBLES VIAJES ATIPICOS")
    print("=" * 60)

    centroides_asignados = modelo.cluster_centers_[etiquetas]
    distancias = np.linalg.norm(X_escalado - centroides_asignados, axis=1)
    umbral = np.percentile(distancias, PERCENTIL_ATIPICOS)

    df_atipicos = df.copy()
    df_atipicos["cluster"] = etiquetas
    df_atipicos["distancia_centroide"] = distancias
    df_atipicos["silhouette_individual"] = silhouette_individual
    df_atipicos["posible_atipico"] = df_atipicos["distancia_centroide"] > umbral

    candidatos = df_atipicos[df_atipicos["posible_atipico"]].copy()

    print("\nUmbral de distancia (percentil 95):")
    print(round(umbral, 4))
    print("\nTotal de viajes atípicos candidatos:", len(candidatos))
    print("Porcentaje:", round(len(candidatos) / len(df_atipicos) * 100, 2), "%")

    df_atipicos.to_csv(f"{CARPETA_RESULTADOS}/p08_analisis_distancias.csv", index=False)
    candidatos.to_csv(f"{CARPETA_RESULTADOS}/p08_posibles_atipicos.csv", index=False)

    plt.figure(figsize=(10, 5))
    plt.scatter(range(len(distancias)), distancias, color="#27ae60", alpha=0.7)
    plt.axhline(y=umbral, color="red", linestyle="--", label=f"Umbral P{PERCENTIL_ATIPICOS}")
    plt.title("Velox UTVT - Distancia de viajes a sus centroides")
    plt.xlabel("Índice del viaje")
    plt.ylabel("Distancia euclidiana")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{CARPETA_RESULTADOS}/p08_distancias_atipicos.png", dpi=300)
    plt.close()

    return df_atipicos, candidatos, umbral

def visualizar_segmentos(df, etiquetas):
    plt.figure(figsize=(10, 6))
    dispersion = plt.scatter(
        df["costo"],
        df["asientos_disponibles"],
        c=etiquetas,
        cmap="Greens",
        alpha=0.8,
        edgecolors="black",
        linewidths=0.5
    )
    plt.title("Velox UTVT - Segmentacion Operativa de Viajes")
    plt.xlabel("Costo del Viaje ($)")
    plt.ylabel("Asientos Disponibles")
    plt.colorbar(dispersion, label="Cluster")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{CARPETA_RESULTADOS}/p08_segmentacion_final.png", dpi=300)
    plt.close()

def exportar_reporte(df, etiquetas, silhouette_individual, df_centroides, candidatos):
    reporte = df.copy()
    reporte["cluster"] = etiquetas
    reporte["silhouette_individual"] = silhouette_individual

    ids_atipicos = set(candidatos["id_viaje"].tolist())
    reporte["posible_atipico"] = reporte["id_viaje"].isin(ids_atipicos)

    orden = df_centroides.sort_values("costo")["cluster"].tolist()
    nombres = {}
    if len(orden) == 3:
        nombres = {
            orden[0]: "Demanda Baja / Económico",
            orden[1]: "Compromiso Intermedio",
            orden[2]: "Alta Demanda / Premium"
        }

    reporte["perfil_operativo"] = reporte["cluster"].map(nombres)
    reporte.to_csv(f"{CARPETA_RESULTADOS}/p08_reporte_integral.csv", index=False)
    print("\nReporte integral exportado correctamente en la carpeta:", CARPETA_RESULTADOS)

def main():
    print("\n" + "=" * 65)
    print("VELOX UTVT - PRACTICA 8")
    print("ANALISIS AVANZADO Y OPTIMIZACION DE SEGMENTOS DE VIAJES")
    print("=" * 65)

    df = cargar_datos()
    df, X = preparar_datos(df)

    resultados_init, modelos_init, X_std, scaler_std = comparar_inicializaciones(X)
    resultados_scaler, modelos_scaler = comparar_escaladores(X)
    df_ari, particiones = analizar_estabilidad(X_std)

    modelo, escalador, X_escalado, etiquetas, silhouette_individual = entrenar_modelo_final(X)
    df_centroides, perfiles = analizar_centroides(df, modelo, escalador, etiquetas)
    df_atipicos, candidatos, umbral = detectar_atipicos(df, X_escalado, modelo, etiquetas, silhouette_individual)

    visualizar_segmentos(df, etiquetas)
    exportar_reporte(df, etiquetas, silhouette_individual, df_centroides, candidatos)

    print("\n" + "=" * 65)
    print("RESUMEN FINAL DE LA PRACTICA 8 (VELOX UTVT)")
    print("=" * 65)
    print("Viajes analizados:", len(df))
    print("Clusters operativos:", NUM_CLUSTERS)
    print("Silhouette final:", round(silhouette_score(X_escalado, etiquetas), 4))
    print("Posibles viajes atípicos detectados:", len(candidatos))
    print("Porcentaje de atípicos:", round(len(candidatos) / len(df) * 100, 2), "%")
    print("\nArchivos guardados exitosamente en:", CARPETA_RESULTADOS)
    print("PRACTICA 8 FINALIZADA CORRECTAMENTE.")

if __name__ == "__main__":
    main()
