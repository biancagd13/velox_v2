import os
import matplotlib
matplotlib.use('Agg')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.templating import Jinja2Templates

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import silhouette_score, silhouette_samples, adjusted_rand_score

from database import db

router = APIRouter(prefix="/ia/practica08", tags=["IA - Práctica 08"])
templates = Jinja2Templates(directory="templates")

os.makedirs("static/img/ia", exist_ok=True)
os.makedirs("resultados_p08_velox", exist_ok=True)
SEMILLA = 42
NUM_CLUSTERS = 3
NUM_INICIALIZACIONES = 20
PERCENTIL_ATIPICOS = 95
VARIABLES = ["costo", "asientos_disponibles"]

@router.get("/", response_class=HTMLResponse)
async def vista_practica08(request: Request):
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

    if "id_viaje" not in df.columns:
        df.insert(0, "id_viaje", [f"VIAJE-{i:04d}" for i in range(1, len(df) + 1)])

    for col in VARIABLES:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    
    # Valores faltantes y preparación
    nulos = df[VARIABLES].isnull().sum().to_dict()
    df[VARIABLES] = df[VARIABLES].fillna(df[VARIABLES].median())
    df["costo"] = df["costo"].clip(10, 500)
    df["asientos_disponibles"] = df["asientos_disponibles"].clip(1, 6)

    X = df[VARIABLES].to_numpy()

    # 1. Comparación de Inicializaciones
    escalador_std = StandardScaler()
    X_std = escalador_std.fit_transform(X)
    
    res_init = []
    for init_type in ["k-means++", "random"]:
        mod = KMeans(n_clusters=NUM_CLUSTERS, init=init_type, n_init=NUM_INICIALIZACIONES, random_state=SEMILLA)
        etiq = mod.fit_predict(X_std)
        sil = silhouette_score(X_std, etiq)
        res_init.append({"inicializacion": init_type, "inercia": round(float(mod.inertia_), 4), "silhouette": round(float(sil), 4)})

    plt.figure(figsize=(6, 3.5))
    df_init = pd.DataFrame(res_init)
    plt.bar(df_init["inicializacion"], df_init["silhouette"], color="#27ae60")
    plt.title("Comparación de Inicializaciones")
    plt.ylabel("Silhouette")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    img_init = "static/img/ia/p08_inicializaciones.png"
    plt.savefig(img_init, dpi=150)
    plt.close()

    # 2. Comparación de Escaladores
    escaladores = {"StandardScaler": StandardScaler(), "MinMaxScaler": MinMaxScaler()}
    res_esc = []
    for nombre, esc in escaladores.items():
        X_trans = esc.fit_transform(X)
        mod = KMeans(n_clusters=NUM_CLUSTERS, init="k-means++", n_init=NUM_INICIALIZACIONES, random_state=SEMILLA)
        etiq = mod.fit_predict(X_trans)
        sil = silhouette_score(X_trans, etiq)
        res_esc.append({"escalador": nombre, "inercia": round(float(mod.inertia_), 4), "silhouette": round(float(sil), 4)})

    plt.figure(figsize=(6, 3.5))
    df_esc = pd.DataFrame(res_esc)
    plt.bar(df_esc["escalador"], df_esc["silhouette"], color="#27ae60")
    plt.title("Comparación de Escaladores")
    plt.ylabel("Silhouette")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    img_esc = "static/img/ia/p08_escaladores.png"
    plt.savefig(img_esc, dpi=150)
    plt.close()

    # 3. Análisis de Estabilidad ARI
    semillas = [10, 20, 30, 40, 50, 42]
    particiones = {}
    for sem in semillas:
        mod = KMeans(n_clusters=NUM_CLUSTERS, init="k-means++", n_init=NUM_INICIALIZACIONES, random_state=sem)
        particiones[sem] = mod.fit_predict(X_std)

    resultados_ari = []
    for i in range(len(semillas)):
        for j in range(i + 1, len(semillas)):
            sem_a = semillas[i]
            sem_b = semillas[j]
            ari_val = adjusted_rand_score(particiones[sem_a], particiones[sem_b])
            resultados_ari.append({"semilla_a": sem_a, "semilla_b": sem_b, "ARI": round(float(ari_val), 4)})

    df_ari = pd.DataFrame(resultados_ari)
    ari_promedio = round(float(df_ari["ARI"].mean()), 4) if not df_ari.empty else 1.0

    # 4. Análisis de Centroides y Modelo Final
    modelo_final = KMeans(n_clusters=NUM_CLUSTERS, init="k-means++", n_init=NUM_INICIALIZACIONES, random_state=SEMILLA)
    etiquetas = modelo_final.fit_predict(X_std)
    silhouette_ind = silhouette_samples(X_std, etiquetas)

    centroides_orig = escalador_std.inverse_transform(modelo_final.cluster_centers_)
    df_cent = pd.DataFrame(centroides_orig, columns=VARIABLES)
    df_cent.insert(0, "cluster", range(NUM_CLUSTERS))

    df_perfiles = df[VARIABLES].copy()
    df_perfiles["cluster"] = etiquetas
    perfiles = df_perfiles.groupby("cluster")[VARIABLES].mean().reset_index().round(2)

    orden = df_cent.sort_values("costo")["cluster"].tolist()
    nombres = {orden[0]: "Demanda Baja / Económico", orden[1]: "Compromiso Intermedio", orden[2]: "Alta Demanda / Premium"}

    df["cluster"] = etiquetas
    df["silhouette_individual"] = silhouette_ind
    df["perfil_operativo"] = df["cluster"].map(nombres)
    df_cent["perfil"] = df_cent["cluster"].map(nombres)

    # 5. Detección de atípicos
    centroides_asignados = modelo_final.cluster_centers_[etiquetas]
    distancias = np.linalg.norm(X_std - centroides_asignados, axis=1)
    umbral = np.percentile(distancias, PERCENTIL_ATIPICOS)
    df["distancia_centroide"] = distancias
    df["posible_atipico"] = df["distancia_centroide"] > umbral

    candidatos = df[df["posible_atipico"]]
    porcentaje_atipicos = round((len(candidatos) / len(df)) * 100, 2)
    silhouette_final = round(float(silhouette_score(X_std, etiquetas)), 4)

    # Gráfica de segmentación final
    plt.figure(figsize=(7, 4))
    disp = plt.scatter(df["costo"], df["asientos_disponibles"], c=etiquetas, cmap="Greens", alpha=0.8, edgecolors="black", linewidths=0.5)
    plt.scatter(centroides_orig[:, 0], centroides_orig[:, 1], marker="X", s=250, color="black", label="Centroides")
    plt.title("Segmentación Avanzada y Centroides")
    plt.xlabel("Costo ($)")
    plt.ylabel("Asientos Disponibles")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    img_seg = "static/img/ia/p08_segmentacion_final.png"
    plt.savefig(img_seg, dpi=150)
    plt.close()

    # Exportar archivos CSV
    df.to_csv("resultados_p08_velox/p08_reporte_integral.csv", index=False, encoding="utf-8-sig")
    candidatos.to_csv("resultados_p08_velox/p08_posibles_atipicos.csv", index=False, encoding="utf-8-sig")
    df_cent.to_csv("resultados_p08_velox/p08_centroides.csv", index=False, encoding="utf-8-sig")
    df_ari.to_csv("resultados_p08_velox/p08_estabilidad_ari.csv", index=False, encoding="utf-8-sig")

    return templates.TemplateResponse(
        request,
        "ia_modulo/practica08.html",
        {
            "request": request,
            "total_registros": len(df),
            "nulos_costo": nulos.get("costo", 0),
            "nulos_asientos": nulos.get("asientos_disponibles", 0),
            "k_final": NUM_CLUSTERS,
            "silhouette_final": silhouette_final,
            "ari_promedio": ari_promedio,
            "total_atipicos": len(candidatos),
            "porcentaje_atipicos": porcentaje_atipicos,
            "umbral_distancia": round(float(umbral), 4),
            "res_init": res_init,
            "res_esc": res_esc,
            "tabla_ari": df_ari.to_dict(orient="records"),
            "centroides": df_cent.round(2).to_dict(orient="records"),
            "primeros_datos": df.head(15).to_dict(orient="records"),
            "img_init": "/" + img_init,
            "img_esc": "/" + img_esc,
            "img_seg": "/" + img_seg
        }
    )

@router.get("/exportar_csv/{tipo}")
async def exportar_csv(tipo: str):
    rutas = {
        "integral": "resultados_p08_velox/p08_reporte_integral.csv",
        "atipicos": "resultados_p08_velox/p08_posibles_atipicos.csv",
        "centroides": "resultados_p08_velox/p08_centroides.csv"
    }
    ruta = rutas.get(tipo, rutas["integral"])
    if os.path.exists(ruta):
        return FileResponse(ruta, media_type="text/csv", filename=os.path.basename(ruta))
    return {"error": "Archivo no encontrado."}