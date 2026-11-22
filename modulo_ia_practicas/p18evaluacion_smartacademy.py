# ============================================================
# PROYECTO: VELOX UTVT
# PRACTICA 18 (ETAPA 9): EVALUACION INTELIGENTE EN PRODUCCION
#
# Archivo:
# p18evaluacion_velox.py
#
# Objetivo:
# Ejecutar inferencia en tiempo real sobre nuevos trayectos
# de carpooling combinando Random Forest, K-Means, PCA y un
# motor de recomendaciones operativas, almacenando cada
# evaluación en MongoDB Atlas.
#
# Ejecución desde terminal:
# streamlit run p18evaluacion_velox.py
# ============================================================

import os
import sys
import uuid
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import streamlit as st
from pymongo import ASCENDING
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

# ============================================================
# 1. CONFIGURACION DE LA PAGINA (PRIMERA INSTRUCCION STREAMLIT)
# ============================================================

st.set_page_config(
    page_title="Velox UTVT - Evaluación Inteligente de Viajes",
    page_icon="🧠",
    layout="wide"
)

# Permitir importar 'database.py' desde la raíz del proyecto Velox UTVT
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from database import db

ESTADOS_EXITOSOS = ["activo", "aceptado", "en_curso", "finalizado"]
COLECCION_ANALISIS = "analisis_viajes"
SEMILLA = 42


# ============================================================
# 2. PREPARACION Y ENTRENAMIENTO CACHEADO DE MODELOS
# ============================================================

@st.cache_resource
def inicializar_modelos_velox():
    """
    Extrae los viajes reales de MongoDB Atlas ('db.viaje'),
    complementa si el historial aún es pequeño y entrena en cascada:
    1. Random Forest (predicción de éxito del viaje)
    2. K-Means + StandardScaler (segmentación operativa)
    3. PCA (proyección dimensional 2D)
    """
    lista_viajes = []
    try:
        cursor = db.viaje.find({}, {"_id": 0, "costo": 1, "asientos_disponibles": 1, "estado": 1})
        for doc in cursor:
            costo = float(doc.get("costo", 0))
            asientos = int(doc.get("asientos_disponibles", 0))
            estado = str(doc.get("estado", "")).lower().strip()
            exitoso = 1 if estado in ESTADOS_EXITOSOS else 0
            lista_viajes.append({
                "costo": costo,
                "asientos_disponibles": asientos,
                "viaje_exitoso": exitoso
            })
    except Exception:
        pass

    df = pd.DataFrame(lista_viajes)

    # Complementar con registros sintéticos calibrados a Velox UTVT si hay pocos datos reales
    if df.empty or len(df) < 50:
        rng = np.random.default_rng(SEMILLA)
        n_extra = 300 - len(df)
        costos_sim = rng.uniform(10, 45, n_extra).round(1)
        asientos_sim = rng.integers(1, 5, n_extra)
        # Regla probabilística realista: tarifas accesibles y buena disponibilidad favorecen el éxito
        prob_sim = np.clip(0.85 - (costos_sim - 12) * 0.018 + (asientos_sim - 1) * 0.05, 0.1, 0.98)
        exito_sim = (rng.random(n_extra) < prob_sim).astype(int)

        df_sim = pd.DataFrame({
            "costo": costos_sim,
            "asientos_disponibles": asientos_sim,
            "viaje_exitoso": exito_sim
        })
        df = pd.concat([df, df_sim], ignore_index=True)

    X = df[["costo", "asientos_disponibles"]].to_numpy(dtype=float)
    y = df["viaje_exitoso"].to_numpy(dtype=int)

    # 1. Clasificador Supervisado (Random Forest)
    modelo_rf = RandomForestClassifier(
        n_estimators=150,
        max_depth=5,
        random_state=SEMILLA
    )
    modelo_rf.fit(X, y)

    # 2. Escalador y Segmentación No Supervisada (K-Means)
    escalador = StandardScaler()
    X_escalado = escalador.fit_transform(X)

    modelo_kmeans = KMeans(
        n_clusters=3,
        init="k-means++",
        n_init=20,
        random_state=SEMILLA
    )
    modelo_kmeans.fit(X_escalado)

    # Ordenar clusters por costo ascendente para asignar etiquetas semánticas consistentes
    centroides_orig = escalador.inverse_transform(modelo_kmeans.cluster_centers_)
    orden_clusters = np.argsort(centroides_orig[:, 0])
    mapa_perfiles = {
        int(orden_clusters[0]): "Económico / Alta Accesibilidad",
        int(orden_clusters[1]): "Tarifa Estándar / Equilibrado",
        int(orden_clusters[2]): "Tarifa Elevada / Baja Demanda"
    }

    # 3. Reducción de Dimensionalidad (PCA 2D)
    modelo_pca = PCA(n_components=2, random_state=SEMILLA)
    modelo_pca.fit(X_escalado)

    return {
        "rf": modelo_rf,
        "escalador": escalador,
        "kmeans": modelo_kmeans,
        "pca": modelo_pca,
        "mapa_perfiles": mapa_perfiles,
        "total_entrenamiento": len(df)
    }


# ============================================================
# 3. MOTOR DE INFERENCIA Y RECOMENDACIONES VELOX UTVT
# ============================================================

def evaluar_trayecto_ia(artefactos, costo, asientos):
    """
    Ejecuta la inferencia completa sobre un viaje propuesto y genera
    recomendaciones operativas personalizadas.
    """
    entrada = np.array([[float(costo), float(asientos)]], dtype=float)
    entrada_esc = artefactos["escalador"].transform(entrada)

    # Predicción Random Forest
    prediccion = int(artefactos["rf"].predict(entrada)[0])
    probabilidades = artefactos["rf"].predict_proba(entrada)[0]
    prob_exito = round(float(probabilidades[1]) * 100, 2)

    # Segmentación K-Means
    cluster_id = int(artefactos["kmeans"].predict(entrada_esc)[0])
    perfil_operativo = artefactos["mapa_perfiles"].get(cluster_id, "Estándar")

    # Proyección PCA
    coords_pca = artefactos["pca"].transform(entrada_esc)[0]
    pc1 = round(float(coords_pca[0]), 4)
    pc2 = round(float(coords_pca[1]), 4)

    # Motor de recomendaciones institucionales Velox UTVT
    recomendaciones = []
    if costo > 30:
        recomendaciones.append(
            "Considera ajustar la tarifa entre $12 y $22 MXN para alinearla con el promedio estudiantil de la UTVT."
        )
    elif costo <= 16:
        recomendaciones.append(
            "Tarifa altamente competitiva; favorece una ocupación rápida en horarios de entrada y salida."
        )

    if asientos == 1:
        recomendaciones.append(
            "Con solo 1 asiento disponible, publica el viaje con al menos 45 minutos de anticipación."
        )
    elif asientos >= 3:
        recomendaciones.append(
            "Excelente capacidad disponible; ideal para rutas troncales como Terminal, Metepec o Lerma hacia Santa María Atarasquillo."
        )

    if prob_exito >= 70:
        recomendaciones.append(
            "Alta probabilidad de consolidación: el trayecto cumple con los parámetros óptimos de la plataforma."
        )
    else:
        recomendaciones.append(
            "Riesgo moderado/alto de no consolidarse: sugiere compartir el enlace del viaje en grupos de tu carrera."
        )

    return {
        "prediccion": prediccion,
        "resultado": "Viaje Viable / Consolidable" if prediccion == 1 else "Riesgo de Baja Demanda",
        "probabilidad_exito": prob_exito,
        "cluster": cluster_id,
        "perfil": perfil_operativo,
        "pc1": pc1,
        "pc2": pc2,
        "recomendaciones": recomendaciones
    }


# ============================================================
# 4. PERSISTENCIA EN MONGODB ATLAS CON INDICES
# ============================================================

def guardar_analisis_mongodb(documento):
    """
    Almacena el documento de evaluación en 'db.analisis_viajes'
    asegurando índices de unicidad y búsqueda por conductor.
    """
    coleccion = db[COLECCION_ANALISIS]

    coleccion.create_index([("id_analisis", ASCENDING)], unique=True, name="idx_id_analisis_unico")
    coleccion.create_index([("id_conductor", ASCENDING)], name="idx_id_conductor")

    coleccion.insert_one(documento)


# ============================================================
# 5. INTERFAZ WEB EN STREAMLIT
# ============================================================

def main():
    st.title("🧠 Velox UTVT — Evaluación Inteligente de Trayectos")
    st.markdown(
        "Simulador predictivo en producción: evalúa la viabilidad de un viaje "
        "mediante **Random Forest + K-Means + PCA** y guarda el diagnóstico en **MongoDB Atlas**."
    )

    with st.spinner("Cargando modelos de Inteligencia Artificial..."):
        artefactos = inicializar_modelos_velox()

    st.sidebar.success(f"Modelos listos ({artefactos['total_entrenamiento']} registros base)")

    # Formulario de captura del trayecto
    with st.form("form_evaluacion_viaje"):
        st.subheader("📋 Parámetros del Trayecto a Evaluar")

        col1, col2 = st.columns(2)

        with col1:
            id_conductor = st.text_input(
                "Matrícula o ID del Conductor:",
                value="222411050",
                help="Identificador institucional del estudiante conductor."
            )
            origen = st.selectbox(
                "Punto de Origen:",
                options=[
                    "Terminal Toluca",
                    "Toluca Centro",
                    "Metepec",
                    "Lerma Centro",
                    "San Mateo Atenco",
                    "Xonacatlán / Toluca-Naucalpan"
                ]
            )
            destino = st.text_input(
                "Destino:",
                value="Campus UTVT (Santa María Atarasquillo)"
            )

        with col2:
            costo = st.number_input(
                "Costo por Pasajero ($ MXN):",
                min_value=5.0,
                max_value=150.0,
                value=16.0,
                step=1.0
            )
            asientos = st.slider(
                "Asientos Disponibles:",
                min_value=1,
                max_value=6,
                value=3
            )
            guardar_bd = st.checkbox(
                "Guardar evaluación en MongoDB Atlas",
                value=True
            )

        enviado = st.form_submit_button("🚀 Evaluar Viabilidad con IA", type="primary")

    if enviado:
        if not id_conductor.strip():
            st.error("Por favor ingresa un identificador o matrícula de conductor.")
            st.stop()

        resultado_ia = evaluar_trayecto_ia(artefactos, costo, asientos)
        id_analisis = f"VELOX-IA-{uuid.uuid4().hex[:8].upper()}"
        fecha_actual = datetime.now(timezone.utc)

        st.divider()
        st.subheader("📊 Diagnóstico del Modelo Predictivo")

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Resultado", "Viable" if resultado_ia["prediccion"] == 1 else "En Riesgo")
        m2.metric("Probabilidad de Éxito", f"{resultado_ia['probabilidad_exito']:.2f}%")
        m3.metric("Perfil Operativo (K-Means)", resultado_ia["perfil"])
        m4.metric("Proyección PCA", f"({resultado_ia['pc1']}, {resultado_ia['pc2']})")

        if resultado_ia["prediccion"] == 1:
            st.success(f"✅ **{resultado_ia['resultado']}** — ID de análisis: `{id_analisis}`")
        else:
            st.warning(f"⚠️ **{resultado_ia['resultado']}** — ID de análisis: `{id_analisis}`")

        st.subheader("💡 Recomendaciones Operativas para el Conductor")
        for rec in resultado_ia["recomendaciones"]:
            st.info(f"• {rec}")

        if guardar_bd:
            documento_atlas = {
                "id_analisis": id_analisis,
                "sistema": "Velox UTVT",
                "etapa": 9,
                "id_conductor": id_conductor.strip(),
                "fecha_registro": fecha_actual,
                "datos_viaje": {
                    "origen": origen,
                    "destino": destino.strip(),
                    "costo": float(costo),
                    "asientos_disponibles": int(asientos)
                },
                "datos_reporte": {
                    "prediccion": resultado_ia["prediccion"],
                    "resultado": resultado_ia["resultado"],
                    "probabilidad_exito": resultado_ia["probabilidad_exito"],
                    "cluster": resultado_ia["cluster"],
                    "perfil": resultado_ia["perfil"],
                    "pca": {"PC1": resultado_ia["pc1"], "PC2": resultado_ia["pc2"]},
                    "recomendaciones": resultado_ia["recomendaciones"]
                }
            }

            try:
                guardar_analisis_mongodb(documento_atlas)
                st.toast("Evaluación almacenada correctamente en MongoDB Atlas.", icon="✅")
            except Exception as error:
                st.error(f"No se pudo guardar en MongoDB Atlas: {error}")


if __name__ == "__main__":
    main()