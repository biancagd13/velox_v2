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
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

from database import db

router = APIRouter(prefix="/ia/practica07", tags=["IA - Práctica 07"])
templates = Jinja2Templates(directory="templates")

os.makedirs("static/img/ia", exist_ok=True)
os.makedirs("resultados", exist_ok=True)
SEMILLA = 42


@router.get("/", response_class=HTMLResponse)
async def vista_practica07(request: Request):
    # Extracción de MongoDB
    lista_viajes = []
    try:
        cursor = db.viaje.find({}, {"_id": 0, "costo": 1, "asientos_disponibles": 1, "estado": 1})
        for documento in cursor:
            costo = float(documento.get("costo", 0))
            asientos = int(documento.get("asientos_disponibles", 0))
            lista_viajes.append({"costo": costo, "asientos_disponibles": asientos})
    except Exception as error:
        print(f"Error al conectar con MongoDB: {error}")
        
    df = pd.DataFrame(lista_viajes) if lista_viajes else pd.DataFrame(columns=["costo", "asientos_disponibles"])
    
    # Respaldo sintético si hay pocos registros
    if df.empty or len(df) < 50:
        rng = np.random.default_rng(SEMILLA)
        n_sinteticos = 500 - len(df) if len(df) < 500 else 0
        if n_sinteticos > 0:
            df_ext = pd.DataFrame({
                "costo": rng.uniform(20, 200, n_sinteticos),
                "asientos_disponibles": rng.integers(1, 6, n_sinteticos)
            })
            df = pd.concat([df, df_ext], ignore_index=True)

    df["costo"] = df["costo"].clip(10, 500)
    df["asientos_disponibles"] = df["asientos_disponibles"].clip(1, 6)

    # Estadísticas descriptivas para la interfaz
    estadisticas_dict = df.describe().round(2).reset_index().rename(columns={"index": "metrica"}).to_dict(orient="records")

    X = df[["costo", "asientos_disponibles"]]
    escalador = StandardScaler()
    X_escalado = escalador.fit_transform(X)

    # Método del codo y Silhouette
    valores_k = range(1, 9)
    inercias = []
    for k in valores_k:
        mod = KMeans(n_clusters=k, random_state=SEMILLA, n_init=20)
        mod.fit(X_escalado)
        inercias.append(mod.inertia_)

    resultados_silhouette = []
    for k in range(2, 9):
        mod = KMeans(n_clusters=k, random_state=SEMILLA, n_init=20)
        etiquetas = mod.fit_predict(X_escalado)
        puntuacion = silhouette_score(X_escalado, etiquetas)
        resultados_silhouette.append(puntuacion)

    # Gráfica Codo
    plt.figure(figsize=(7, 4))
    plt.plot(list(valores_k), inercias, marker="o", color="#27ae60", linewidth=2)
    plt.title("Velox UTVT - Método del Codo")
    plt.xlabel("Cantidad de grupos (K)")
    plt.ylabel("Inercia (WCSS)")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    img_codo = "static/img/ia/p07_metodo_del_codo.png"
    plt.savefig(img_codo, dpi=150)
    plt.close()

    # Gráfica Silhouette
    plt.figure(figsize=(7, 4))
    plt.plot(list(range(2, 9)), resultados_silhouette, marker="o", color="#27ae60", linewidth=2)
    plt.title("Velox UTVT - Evaluación Silhouette")
    plt.xlabel("Cantidad de grupos (K)")
    plt.ylabel("Silhouette Score")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    img_sil = "static/img/ia/p07_silhouette.png"
    plt.savefig(img_sil, dpi=150)
    plt.close()

    # Modelo final K=3
    K_FINAL = 3
    modelo_final = KMeans(n_clusters=K_FINAL, random_state=SEMILLA, n_init=20)
    df["cluster"] = modelo_final.fit_predict(X_escalado)

    # Perfiles operativos
    perfiles = df.groupby("cluster")[["costo", "asientos_disponibles"]].mean().round(2)
    orden = perfiles["costo"].sort_values().index.tolist()
    nombres = ["Demanda Baja / Económico", "Compromiso Intermedio", "Alta Demanda / Premium"]
    mapa_nombres = {cluster: nombres[i % len(nombres)] for i, cluster in enumerate(orden)}
    df["perfil"] = df["cluster"].map(mapa_nombres)

    # Conteo por grupo
    conteo_grupos = df["cluster"].value_counts().sort_index().to_dict()

    # Centroides originales
    centroides_orig = escalador.inverse_transform(modelo_final.cluster_centers_)
    centroides_df = pd.DataFrame(centroides_orig, columns=X.columns).round(2)
    centroides_df["cluster"] = centroides_df.index
    centroides_df["perfil"] = centroides_df["cluster"].map(mapa_nombres)
    centroides_list = centroides_df.to_dict(orient="records")

    # Gráfica Segmentación
    plt.figure(figsize=(7, 4))
    for cluster in sorted(df["cluster"].unique()):
        datos_cluster = df[df["cluster"] == cluster]
        plt.scatter(datos_cluster["costo"], datos_cluster["asientos_disponibles"], label=mapa_nombres[cluster], alpha=0.65)

    plt.scatter(centroides_orig[:, 0], centroides_orig[:, 1], marker="X", s=250, edgecolor="black", linewidth=1.5, color="black", label="Centroides")
    plt.title("Velox UTVT - Segmentación de Viajes")
    plt.xlabel("Costo del Viaje ($)")
    plt.ylabel("Asientos Disponibles")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    img_seg = "static/img/ia/p07_segmentacion_viajes.png"
    plt.savefig(img_seg, dpi=150)
    plt.close()

    silhouette_final = silhouette_score(X_escalado, df["cluster"])

    # Guardar CSVs
    df.to_csv("resultados/p07_viajes_segmentados.csv", index=False, encoding="utf-8-sig")
    perfiles.to_csv("resultados/p07_perfiles_operativos.csv", encoding="utf-8-sig")

    return templates.TemplateResponse(
        request,
        "ia_modulo/practica07.html",
        {
            "request": request,
            "total_registros": len(df),
            "k_final": K_FINAL,
            "inercia_final": round(modelo_final.inertia_, 2),
            "silhouette_final": round(silhouette_final, 4),
            "estadisticas": estadisticas_dict,
            "centroides": centroides_list,
            "primeros_datos": df.head(15).to_dict(orient="records"),
            "img_codo": "/" + img_codo,
            "img_sil": "/" + img_sil,
            "img_seg": "/" + img_seg
        }
    )


@router.get("/exportar_csv/{tipo}")
async def exportar_csv(tipo: str):
    rutas = {
        "segmentados": "resultados/p07_viajes_segmentados.csv",
        "perfiles": "resultados/p07_perfiles_operativos.csv"
    }
    ruta = rutas.get(tipo, rutas["segmentados"])
    if os.path.exists(ruta):
        return FileResponse(ruta, media_type="text/csv", filename=os.path.basename(ruta))
    return {"error": "Archivo no encontrado."}