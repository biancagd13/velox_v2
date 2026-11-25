# ============================================================
# PROYECTO: VELOX UTVT
# PRACTICA 9: REDUCCION DE DIMENSIONALIDAD MEDIANTE PCA
# (Principal Component Analysis o Análisis de Componentes Principales)
# ============================================================

import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from database import db

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import mean_squared_error

SEMILLA = 42
NUM_CLUSTERS = 3
VARIANZA_OBJETIVO = 0.85
CARPETA_RESULTADOS = "resultados_p09_velox"

os.makedirs(CARPETA_RESULTADOS, exist_ok=True)
os.makedirs("static/img/ia", exist_ok=True)

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

    print("\nDatos cargados correctamente.")
    print("Total de viajes analizados:", len(df))
    return df

def preparar_datos(df):
    X = df[VARIABLES].copy()
    for columna in VARIABLES:
        X[columna] = pd.to_numeric(X[columna], errors="coerce")
    X = X.fillna(X.median())
    X["costo"] = X["costo"].clip(10, 500)
    X["asientos_disponibles"] = X["asientos_disponibles"].clip(1, 6)
    return X.to_numpy(dtype=float)

def estandarizar_datos(X):
    escalador = StandardScaler()
    X_escalado = escalador.fit_transform(X)

    print("\n" + "=" * 60)
    print("ESTANDARIZACION")
    print("=" * 60)
    print("\nMedia de las variables escaladas:")
    print(np.round(X_escalado.mean(axis=0), 4))
    print("\nDesviacion estandar:")
    print(np.round(X_escalado.std(axis=0), 4))

    return X_escalado, escalador

def obtener_clusters(df, X_escalado):
    if "cluster" in df.columns:
        print("\nSe utilizaran los clusters previos.")
        etiquetas = df["cluster"].to_numpy()
    else:
        print("\nGenerando segmentos operativos con K-Means...")
        modelo = KMeans(
            n_clusters=NUM_CLUSTERS,
            init="k-means++",
            n_init=20,
            random_state=SEMILLA
        )
        etiquetas = modelo.fit_predict(X_escalado)

    return np.asarray(etiquetas).astype(int)

def entrenar_pca(X_escalado):
    n_comp = X_escalado.shape[1]
    modelo_pca = PCA(n_components=n_comp)
    datos_pca = modelo_pca.fit_transform(X_escalado)

    print("\n" + "=" * 60)
    print("RESULTADOS GENERALES DE PCA")
    print("=" * 60)
    print("\nVarianza explicada por componente:")
    print(np.round(modelo_pca.explained_variance_ratio_, 4))
    print("\nVarianza explicada en porcentaje:")
    print(np.round(modelo_pca.explained_variance_ratio_ * 100, 2))

    varianza_acumulada = np.cumsum(modelo_pca.explained_variance_ratio_)
    print("\nVarianza acumulada:")
    print(np.round(varianza_acumulada * 100, 2))

    return modelo_pca, datos_pca, varianza_acumulada

def generar_tabla_varianza(modelo_pca):
    varianza = modelo_pca.explained_variance_ratio_
    acumulada = np.cumsum(varianza)
    n_comp = len(varianza)

    df_varianza = pd.DataFrame({
        "componente": [f"PC{i+1}" for i in range(n_comp)],
        "varianza_explicada": varianza,
        "porcentaje_explicado": varianza * 100,
        "varianza_acumulada": acumulada,
        "porcentaje_acumulado": acumulada * 100
    })

    print("\nTabla de varianza explicada:")
    print(df_varianza.round(4).to_string(index=False))

    df_varianza.to_csv(f"{CARPETA_RESULTADOS}/p09_varianza_explicada.csv", index=False)
    return df_varianza

def analizar_cargas(modelo_pca):
    cargas = pd.DataFrame(
        modelo_pca.components_,
        columns=VARIABLES,
        index=[f"PC{i+1}" for i in range(modelo_pca.n_components_)]
    )

    print("\n" + "=" * 60)
    print("CARGAS DE LAS COMPONENTES")
    print("=" * 60)
    print(cargas.round(4))

    cargas.to_csv(f"{CARPETA_RESULTADOS}/p09_cargas_componentes.csv")

    plt.figure(figsize=(8, 4))
    plt.imshow(cargas.values, aspect="auto", cmap="Greens", vmin=-1, vmax=1)
    plt.colorbar(label="Carga de componente")
    plt.xticks(range(len(VARIABLES)), VARIABLES, rotation=25, ha="right")
    plt.yticks(range(modelo_pca.n_components_), cargas.index)
    plt.title("Velox UTVT - Cargas de las componentes principales")
    plt.tight_layout()
    plt.savefig(f"{CARPETA_RESULTADOS}/p09_cargas_componentes.png", dpi=300)
    plt.close()

    return cargas

def determinar_componentes(modelo_pca):
    acumulada = np.cumsum(modelo_pca.explained_variance_ratio_)
    cantidad = np.argmax(acumulada >= VARIANZA_OBJETIVO) + 1
    if acumulada[-1] < VARIANZA_OBJETIVO:
        cantidad = len(acumulada)

    print("\n" + "=" * 60)
    print("SELECCION DE COMPONENTES")
    print("=" * 60)
    print("Varianza objetivo:", f"{VARIANZA_OBJETIVO * 100:.2f}%")
    print("Componentes necesarias:", cantidad)
    print("Varianza acumulada conservada:", round(acumulada[cantidad - 1] * 100, 2), "%")

    return cantidad

def graficar_varianza(df_varianza):
    plt.figure(figsize=(8, 5))
    plt.plot(df_varianza["componente"], df_varianza["porcentaje_explicado"], marker="o", color="#27ae60", linewidth=2)
    plt.title("Velox UTVT - Gráfica de sedimentación PCA")
    plt.xlabel("Componentes principales")
    plt.ylabel("Varianza explicada (%)")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{CARPETA_RESULTADOS}/p09_sedimentacion.png", dpi=300)
    plt.close()

def visualizar_pca_2d(datos_pca, etiquetas):
    plt.figure(figsize=(9, 6))
    dispersion = plt.scatter(
        datos_pca[:, 0],
        datos_pca[:, 1],
        c=etiquetas,
        cmap="Greens",
        alpha=0.8,
        s=50,
        edgecolors="black",
        linewidths=0.5
    )
    plt.title("Velox UTVT - Proyeccion PCA en 2D")
    plt.xlabel("Componente principal 1 (PC1)")
    plt.ylabel("Componente principal 2 (PC2)")
    plt.colorbar(dispersion, label="Cluster")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{CARPETA_RESULTADOS}/p09_pca_2d.png", dpi=300)
    plt.savefig("static/img/ia/p09_pca_2d.png", dpi=150)
    plt.close()

def evaluar_reconstruccion(X_escalado, datos_pca, modelo_pca, cantidad_componentes):
    datos_reducidos = datos_pca[:, :cantidad_componentes]
    reconstruidos = (
        datos_reducidos @ modelo_pca.components_[:cantidad_componentes, :]
    ) + modelo_pca.mean_

    error = mean_squared_error(X_escalado, reconstruidos)

    print("\n" + "=" * 60)
    print("EVALUACION DE RECONSTRUCCION")
    print("=" * 60)
    print("Componentes utilizadas:", cantidad_componentes)
    print("Error cuadratico medio (MSE):", round(error, 6))

    return reconstruidos, error

def exportar_coordenadas(df, datos_pca, etiquetas):
    resultado = pd.DataFrame({
        "id_viaje": df["id_viaje"],
        "cluster": etiquetas,
        "PC1": datos_pca[:, 0],
        "PC2": datos_pca[:, 1]
    })
    resultado.to_csv(f"{CARPETA_RESULTADOS}/p09_coordenadas_pca.csv", index=False, encoding="utf-8-sig")
    print("\nCoordenadas PCA exportadas correctamente.")
    return resultado

def main():
    print("\n" + "=" * 65)
    print("VELOX UTVT - PRACTICA 9")
    print("REDUCCION DE DIMENSIONALIDAD MEDIANTE PCA")
    print("=" * 65)

    df = cargar_datos()
    X = preparar_datos(df)
    X_escalado, escalador = estandarizar_datos(X)
    etiquetas = obtener_clusters(df, X_escalado)

    modelo_pca, datos_pca, acumulada = entrenar_pca(X_escalado)
    df_varianza = generar_tabla_varianza(modelo_pca)
    cargas = analizar_cargas(modelo_pca)
    cantidad_componentes = determinar_componentes(modelo_pca)

    graficar_varianza(df_varianza)
    visualizar_pca_2d(datos_pca, etiquetas)

    reconstruidos, error = evaluar_reconstruccion(X_escalado, datos_pca, modelo_pca, cantidad_componentes)
    coordenadas = exportar_coordenadas(df, datos_pca, etiquetas)

    print("\n" + "=" * 65)
    print("RESUMEN FINAL DE LA PRACTICA 9 (VELOX UTVT)")
    print("=" * 65)
    print("Viajes analizados:", len(df))
    print("Variables originales:", len(VARIABLES))
    print("Componentes principales:", len(modelo_pca.components_))
    print("Varianza de PC1:", round(modelo_pca.explained_variance_ratio_[0] * 100, 2), "%")
    print("Varianza acumulada PC1 y PC2:", round(acumulada[-1] * 100, 2), "%")
    print("Componentes seleccionadas:", cantidad_componentes)
    print("Error de reconstruccion:", round(error, 6))
    print("\nArchivos generados en la carpeta:", CARPETA_RESULTADOS)
    print("PRACTICA 9 FINALIZADA CORRECTAMENTE.")

if __name__ == "__main__":
    main()
