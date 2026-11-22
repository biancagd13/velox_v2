# ============================================================
# PROYECTO: VELOX UTVT
# PRACTICA 19 (ETAPA 11): REGISTRO OPERATIVO INTEGRAL MULTI-COLECCION
#
# Archivo:
# p19registro_integral_velox.py
#
# Objetivo:
# Gestionar el alta de conductores, captura múltiple de rutas
# mediante tabla editable (st.data_editor), evaluación en lote
# con Inteligencia Artificial y persistencia referenciada en
# tres colecciones de MongoDB Atlas:
#   1. conductores_velox
#   2. rutas_evaluadas
#   3. analisis_viajes
#
# Ejecución desde terminal:
# streamlit run p19registro_integral_velox.py --server.headless true
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
# 1. CONFIGURACION DE PAGINA (PRIMERA INSTRUCCION STREAMLIT)
# ============================================================

st.set_page_config(
    page_title="Velox UTVT - Registro Operativo Integral",
    page_icon="🗂️",
    layout="wide"
)

# Permitir importar 'database.py' desde la raíz del proyecto Velox UTVT
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from database import db

ESTADOS_EXITOSOS = ["activo", "aceptado", "en_curso", "finalizado"]
SEMILLA = 42


# ============================================================
# 2. ENTRENAMIENTO CACHEADO DE MODELOS BASE
# ============================================================

@st.cache_resource
def cargar_modelos_velox():
    """
    Entrena los modelos de clasificación (Random Forest),
    segmentación (K-Means) y proyección (PCA) con datos de 'db.viaje'
    complementados con respaldo sintético calibrado.
    """
    lista_viajes = []
    try:
        cursor = db.viaje.find({}, {"_id": 0, "costo": 1, "asientos_disponibles": 1, "estado": 1})
        for doc in cursor:
            costo = float(doc.get("costo", 0))
            asientos = int(doc.get("asientos_disponibles", 0))
            estado = str(doc.get("estado", "")).lower().strip()
            lista_viajes.append({
                "costo": costo,
                "asientos_disponibles": asientos,
                "viaje_exitoso": 1 if estado in ESTADOS_EXITOSOS else 0
            })
    except Exception:
        pass

    df = pd.DataFrame(lista_viajes)

    if df.empty or len(df) < 50:
        rng = np.random.default_rng(SEMILLA)
        n_extra = 300 - len(df)
        costos_sim = rng.uniform(10, 45, n_extra).round(1)
        asientos_sim = rng.integers(1, 5, n_extra)
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

    rf = RandomForestClassifier(n_estimators=150, max_depth=5, random_state=SEMILLA).fit(X, y)

    escalador = StandardScaler()
    X_esc = escalador.fit_transform(X)

    kmeans = KMeans(n_clusters=3, init="k-means++", n_init=20, random_state=SEMILLA).fit(X_esc)
    centroides_orig = escalador.inverse_transform(kmeans.cluster_centers_)
    orden = np.argsort(centroides_orig[:, 0])
    mapa_perfiles = {
        int(orden[0]): "Económico / Alta Accesibilidad",
        int(orden[1]): "Tarifa Estándar / Equilibrado",
        int(orden[2]): "Tarifa Elevada / Baja Demanda"
    }

    pca = PCA(n_components=2, random_state=SEMILLA).fit(X_esc)

    return {
        "rf": rf,
        "escalador": escalador,
        "kmeans": kmeans,
        "pca": pca,
        "mapa_perfiles": mapa_perfiles
    }


# ============================================================
# 3. EVALUACION INDIVIDUAL DE CADA RUTA CAPTURADA
# ============================================================

def evaluar_ruta(artefactos, costo, asientos):
    """
    Calcula predicción, probabilidad de éxito, perfil K-Means y PCA
    para una ruta individual dentro de la tabla del conductor.
    """
    entrada = np.array([[float(costo), float(asientos)]], dtype=float)
    entrada_esc = artefactos["escalador"].transform(entrada)

    pred = int(artefactos["rf"].predict(entrada)[0])
    prob = round(float(artefactos["rf"].predict_proba(entrada)[0][1]) * 100, 2)
    cluster = int(artefactos["kmeans"].predict(entrada_esc)[0])
    perfil = artefactos["mapa_perfiles"].get(cluster, "Estándar")
    coords = artefactos["pca"].transform(entrada_esc)[0]

    return {
        "prediccion": pred,
        "resultado": "Viable / Consolidable" if pred == 1 else "Riesgo Operativo",
        "probabilidad_exito": prob,
        "cluster": cluster,
        "perfil": perfil,
        "pc1": round(float(coords[0]), 4),
        "pc2": round(float(coords[1]), 4)
    }


# ============================================================
# 4. PERSISTENCIA INTEGRAL EN 3 COLECCIONES (ETAPA 11)
# ============================================================

def asegurar_indices():
    """Crea índices únicos y de referencia en las 3 colecciones."""
    db.conductores_velox.create_index([("id_conductor", ASCENDING)], unique=True)
    db.rutas_evaluadas.create_index([("id_evaluacion_rutas", ASCENDING)], unique=True)
    db.rutas_evaluadas.create_index([("id_conductor", ASCENDING)])
    db.analisis_viajes.create_index([("id_analisis", ASCENDING)], unique=True)
    db.analisis_viajes.create_index([("id_evaluacion_rutas", ASCENDING)])


def guardar_registro_integral(datos_conductor, df_rutas, resultados_ia):
    """
    Persiste la operación completa en MongoDB Atlas:
      1. conductores_velox (upsert con $setOnInsert)
      2. rutas_evaluadas (detalle de rutas y promedios)
      3. analisis_viajes (diagnóstico IA vinculado por llave lógica)
    """
    asegurar_indices()
    ahora = datetime.now(timezone.utc)
    id_conductor = datos_conductor["id_conductor"]
    id_eval_rutas = f"EVAL-RUTAS-{uuid.uuid4().hex[:8].upper()}"

    # 1. Documento maestro del conductor (upsert)
    db.conductores_velox.update_one(
        {"id_conductor": id_conductor},
        {
            "$set": {
                "nombre": datos_conductor["nombre"],
                "carrera": datos_conductor["carrera"],
                "vehiculo": datos_conductor["vehiculo"],
                "ultima_actualizacion": ahora
            },
            "$setOnInsert": {
                "fecha_alta": ahora,
                "sistema": "Velox UTVT"
            }
        },
        upsert=True
    )

    # 2. Documento de rutas evaluadas
    lista_rutas_doc = df_rutas.to_dict(orient="records")
    costo_promedio = round(float(df_rutas["costo"].mean()), 2)
    asientos_promedio = round(float(df_rutas["asientos_disponibles"].mean()), 2)

    doc_rutas = {
        "id_evaluacion_rutas": id_eval_rutas,
        "id_conductor": id_conductor,
        "fecha_registro": ahora,
        "total_rutas": len(lista_rutas_doc),
        "costo_promedio": costo_promedio,
        "asientos_promedio": asientos_promedio,
        "rutas": lista_rutas_doc
    }
    db.rutas_evaluadas.insert_one(doc_rutas)

    # 3. Documentos de análisis IA vinculados
    docs_analisis = []
    for ruta_row, res_ia in zip(lista_rutas_doc, resultados_ia):
        id_analisis = f"VELOX-IA-{uuid.uuid4().hex[:8].upper()}"
        docs_analisis.append({
            "id_analisis": id_analisis,
            "id_evaluacion_rutas": id_eval_rutas,
            "id_conductor": id_conductor,
            "sistema": "Velox UTVT",
            "etapa": 11,
            "fecha_registro": ahora,
            "datos_viaje": ruta_row,
            "datos_reporte": res_ia
        })

    if docs_analisis:
        db.analisis_viajes.insert_many(docs_analisis)

    return id_eval_rutas, len(docs_analisis)


# ============================================================
# 5. INTERFAZ WEB CON DATA EDITOR EN STREAMLIT
# ============================================================

def main():
    st.title("🗂️ Velox UTVT — Registro Operativo Integral (Multi-Colección)")
    st.markdown(
        "Alta de conductor, captura editable de múltiples trayectos y evaluación "
        "en lote con Inteligencia Artificial persistida en **MongoDB Atlas**[cite: 19]."
    )

    artefactos = cargar_modelos_velox()

    # --------------------------------------------------------
    # SECCION 1: DATOS DEL CONDUCTOR
    # --------------------------------------------------------
    st.subheader("1. Perfil del Conductor Universitario")
    c1, c2, c3, c4 = st.columns(4)

    with c1:
        id_conductor = st.text_input("Matrícula / ID Conductor:", value="222411050")
    with c2:
        nombre = st.text_input("Nombre Completo:", value="Bianca Gutiérrez Díaz")
    with c3:
        carrera = st.selectbox(
            "Programa Educativo:",
            [
                "Ing. Desarrollo de Software Multiplataforma",
                "Ing. Mecatrónica",
                "Ing. Tecnologías de la Información",
                "Lic. Negocios y Mercadotecnia",
                "Otro"
            ]
        )
    with c4:
        vehiculo = st.text_input("Vehículo / Referencia:", value="Sedán Gris - Placas MEX")

    st.divider()

    # --------------------------------------------------------
    # SECCION 2: TABLA EDITABLE DE RUTAS (st.data_editor)
    # --------------------------------------------------------
    st.subheader("2. Configuración de Rutas Ofertas (Tabla Editable)")
    st.caption("Agrega, edita o elimina filas directamente antes de procesar la evaluación.")

    plantilla_rutas = pd.DataFrame([
        {"origen": "Terminal Toluca", "destino": "Campus UTVT", "costo": 16.0, "asientos_disponibles": 3, "horario": "07:00"},
        {"origen": "Metepec Centro", "destino": "Campus UTVT", "costo": 15.0, "asientos_disponibles": 2, "horario": "07:15"},
        {"origen": "Lerma", "destino": "Campus UTVT", "costo": 12.0, "asientos_disponibles": 3, "horario": "07:30"}
    ])

    df_editado = st.data_editor(
        plantilla_rutas,
        num_rows="dynamic",
        width="stretch",
        column_config={
            "origen": st.column_config.TextColumn("Punto de Origen", required=True),
            "destino": st.column_config.TextColumn("Destino", default="Campus UTVT", required=True),
            "costo": st.column_config.NumberColumn("Costo ($ MXN)", min_value=5.0, max_value=200.0, step=1.0, format="$%.2f"),
            "asientos_disponibles": st.column_config.NumberColumn("Asientos", min_value=1, max_value=6, step=1),
            "horario": st.column_config.TextColumn("Horario Salida", default="07:00")
        }
    )

    if st.button("🚀 Evaluar Rutas y Guardar Registro Integral", type="primary"):
        if not id_conductor.strip() or not nombre.strip():
            st.error("Completa la matrícula y el nombre del conductor.")
            st.stop()

        df_limpio = df_editado.dropna(subset=["origen", "costo", "asientos_disponibles"]).copy()
        if df_limpio.empty:
            st.warning("Debes capturar al menos una ruta válida en la tabla.")
            st.stop()

        # Evaluar cada fila con los modelos de IA
        resultados_ia = []
        for _, fila in df_limpio.iterrows():
            res = evaluar_ruta(artefactos, fila["costo"], fila["asientos_disponibles"])
            resultados_ia.append(res)

        df_resumen = df_limpio.copy()
        df_resumen["Prob. Éxito (%)"] = [r["probabilidad_exito"] for r in resultados_ia]
        df_resumen["Diagnóstico IA"] = [r["resultado"] for r in resultados_ia]
        df_resumen["Perfil K-Means"] = [r["perfil"] for r in resultados_ia]

        st.subheader("3. Resultados de Evaluación por Ruta")
        st.dataframe(df_resumen, width="stretch")

        # Guardar en las 3 colecciones de MongoDB Atlas
        try:
            datos_cond = {
                "id_conductor": id_conductor.strip(),
                "nombre": nombre.strip(),
                "carrera": carrera,
                "vehiculo": vehiculo.strip()
            }
            id_eval, total_docs = guardar_registro_integral(datos_cond, df_limpio, resultados_ia)
            st.success(
                f"✅ Registro integral completado en MongoDB Atlas. "
                f"Folio de rutas: `{id_eval}` | Evaluaciones IA vinculadas: **{total_docs}**."
            )
        except Exception as error:
            st.error(f"Error al persistir el registro integral en MongoDB Atlas: {error}")


if __name__ == "__main__":
    main()