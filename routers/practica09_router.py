import os
import matplotlib
matplotlib.use('Agg')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.templating import Jinja2Templates

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import mean_squared_error

from database import db

router = APIRouter(prefix="/ia/practica09", tags=["IA - Práctica 09"])
templates = Jinja2Templates(directory="templates")

os.makedirs("static/img/ia", exist_ok=True)
os.makedirs("resultados_p09_velox", exist_ok=True)

SEMILLA = 42
NUM_CLUSTERS = 3
VARIANZA_OBJETIVO = 0.85
VARIABLES = ["costo", "asientos_disponibles"]

@router.get("/", response_class=HTMLResponse)
async def vista_practica09(request: Request):
    # Carga de datos de MongoDB o respaldo sintético
    lista_viajes = []
    try:
        cursor = db.viaje.find({}, {"_id": 0, "costo": 1, "asientos_disponibles": 1, "estado": 1})
        for documento in cursor:
            costo = float(documento.get("costo", 0))
            asientos = int(documento.get("asientos_disponibles", 0))
            lista_viajes.append({"costo": costo, "asientos_disponibles": asientos})
    except Exception as error:
        print(f"Error al conectar con MongoDB: {error}")

    df = pd.DataFrame(lista_viajes) if lista_viajes else pd.DataFrame(columns=VARIABLES)

    if df.empty or len(df) < 50:
        rng = np.random.default_rng(SEMILLA)
        n_sinteticos = 500 - len(df) if len(df) < 500 else 0
        if n_sinteticos > 0:
            df_ext = pd.DataFrame({
                "costo": rng.uniform(20, 300, n_sinteticos),
                "asientos_disponibles": rng.integers(1, 6, n_sinteticos)
            })
            df = pd.concat([df, df_ext], ignore_index=True)

    if "id_estudiante" not in df.columns:
        df.insert(0, "id_estudiante", [f"EST-{i:04d}" for i in range(1, len(df) + 1)])

    for col in VARIABLES:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    
    df[VARIABLES] = df[VARIABLES].fillna(df[VARIABLES].median())
    df["costo"] = df["costo"].clip(10, 500)
    df["asientos_disponibles"] = df["asientos_disponibles"].clip(1, 6)

    X = df[VARIABLES].to_numpy(dtype=float)

    # 1. Estandarización
    escalador = StandardScaler()
    X_escalado = escalador.fit_transform(X)
    medias_esc = np.round(X_escalado.mean(axis=0), 4).tolist()
    stds_esc = np.round(X_escalado.std(axis=0), 4).tolist()

    # 2. Clusters (K-Means)
    modelo_km = KMeans(n_clusters=NUM_CLUSTERS, init="k-means++", n_init=20, random_state=SEMILLA)
    etiquetas = modelo_km.fit_predict(X_escalado)

    # 3. Entrenamiento PCA
    n_comp = X_escalado.shape[1]
    modelo_pca = PCA(n_components=n_comp)
    datos_pca = modelo_pca.fit_transform(X_escalado)
    
    varianza = modelo_pca.explained_variance_ratio_
    acumulada = np.cumsum(varianza)

    # Tabla Varianza Explicada
    df_varianza = pd.DataFrame({
        "componente": [f"PC{i+1}" for i in range(n_comp)],
        "varianza_explicada": np.round(varianza, 4),
        "porcentaje_explicado": np.round(varianza * 100, 2),
        "varianza_acumulada": np.round(acumulada, 4),
        "porcentaje_acumulado": np.round(acumulada * 100, 2)
    })
    df_varianza.to_csv("resultados_p09_velox/p09_varianza_explicada.csv", index=False)

    # Cargas (Loadings)
    cargas = pd.DataFrame(
        modelo_pca.components_,
        columns=VARIABLES,
        index=[f"PC{i+1}" for i in range(n_comp)]
    )
    cargas.to_csv("resultados_p09_velox/p09_cargas_componentes.csv")

    # Gráfica de Cargas
    plt.figure(figsize=(7, 3.5))
    plt.imshow(cargas.values, aspect="auto", cmap="Greens", vmin=-1, vmax=1)
    plt.colorbar(label="Carga")
    plt.xticks(range(len(VARIABLES)), VARIABLES, rotation=20)
    plt.yticks(range(n_comp), cargas.index)
    plt.title("Cargas de Componentes Principales")
    plt.tight_layout()
    img_cargas = "static/img/ia/p09_cargas.png"
    plt.savefig(img_cargas, dpi=150)
    plt.close()

    # Componentes necesarias para objetivo
    cantidad_componentes = int(np.argmax(acumulada >= VARIANZA_OBJETIVO) + 1)
    if acumulada[-1] < VARIANZA_OBJETIVO:
        cantidad_componentes = len(acumulada)

    # Gráfica sedimentación
    plt.figure(figsize=(6, 3.5))
    plt.plot(df_varianza["componente"], df_varianza["porcentaje_explicado"], marker="o", color="#27ae60", linewidth=2)
    plt.title("Sedimentación PCA")
    plt.xlabel("Componentes")
    plt.ylabel("Varianza Explicada (%)")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    img_sed = "static/img/ia/p09_sedimentacion.png"
    plt.savefig(img_sed, dpi=150)
    plt.close()

    # Gráfica 2D PCA
    plt.figure(figsize=(7, 4.5))
    disp = plt.scatter(datos_pca[:, 0], datos_pca[:, 1], c=etiquetas, cmap="Greens", alpha=0.8, s=50, edgecolors="black", linewidths=0.5)
    plt.title("Proyección PCA en 2D")
    plt.xlabel("Componente Principal 1 (PC1)")
    plt.ylabel("Componente Principal 2 (PC2)")
    plt.colorbar(disp, label="Cluster")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    img_2d = "static/img/ia/p09_pca_2d.png"
    plt.savefig(img_2d, dpi=150)
    plt.close()

    # Error de reconstrucción MSE
    datos_reducidos = datos_pca[:, :cantidad_componentes]
    reconstruidos = (datos_reducidos @ modelo_pca.components_[:cantidad_componentes, :]) + modelo_pca.mean_
    error_mse = float(mean_squared_error(X_escalado, reconstruidos))

    # Exportar coordenadas
    df_coord = pd.DataFrame({
        "id_estudiante": df["id_estudiante"],
        "cluster": etiquetas,
        "PC1": datos_pca[:, 0],
        "PC2": datos_pca[:, 1]
    })
    df_coord.to_csv("resultados_p09_velox/p09_coordenadas_pca.csv", index=False, encoding="utf-8-sig")

    return templates.TemplateResponse(
        request,
        "ia_modulo/practica09.html",
        {
            "request": request,
            "total_registros": len(df),
            "var_pc1": float(round(varianza[0] * 100, 2)),
            "var_acumulada": float(round(acumulada[-1] * 100, 2)),
            "cantidad_componentes": cantidad_componentes,
            "error_mse": round(error_mse, 6),
            "medias_esc": medias_esc,
            "stds_esc": stds_esc,
            "tabla_varianza": df_varianza.to_dict(orient="records"),
            "tabla_cargas": cargas.round(4).reset_index().to_dict(orient="records"),
            "primeras_coordenadas": df_coord.head(15).to_dict(orient="records"),
            "img_cargas": "/" + img_cargas,
            "img_sed": "/" + img_sed,
            "img_2d": "/" + img_2d
        }
    )

@router.get("/exportar_csv/{tipo}")
async def exportar_csv(tipo: str):
    rutas = {
        "coordenadas": "resultados_p09_velox/p09_coordenadas_pca.csv",
        "varianza": "resultados_p09_velox/p09_varianza_explicada.csv",
        "cargas": "resultados_p09_velox/p09_cargas_componentes.csv"
    }
    ruta = rutas.get(tipo, rutas["coordenadas"])
    if os.path.exists(ruta):
        return FileResponse(ruta, media_type="text/csv", filename=os.path.basename(ruta))
    return {"error": "Archivo no encontrado."}