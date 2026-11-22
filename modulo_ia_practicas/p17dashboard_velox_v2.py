# ============================================================
# PROYECTO: VELOX UTVT
# PRACTICA 17 v2 (ETAPA 10): DASHBOARD ANALITICO LONGITUDINAL
#
# Archivo:
# p17dashboard_velox_v2.py
#
# Objetivo:
# Evolucionar el dashboard analítico para incorporar seguimiento
# longitudinal por conductor/usuario, evolución cronológica de
# tarifas y ocupación, y análisis comparativo de viabilidad.
#
# Ejecución desde terminal:
# streamlit run p17dashboard_velox_v2.py
# ============================================================

import os
import sys
from datetime import datetime
import pandas as pd
import plotly.express as px
import streamlit as st

# Permitir importar 'database.py' desde la raíz del proyecto Velox UTVT
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from database import db


# ============================================================
# 1. CONFIGURACION DE LA PAGINA
# ============================================================

st.set_page_config(
    page_title="Velox UTVT - Dashboard Longitudinal v2",
    page_icon="🚗",
    layout="wide"
)

ESTADOS_EXITOSOS = ["activo", "aceptado", "en_curso", "finalizado"]


# ============================================================
# 2. CARGA Y NORMALIZACION DE DATOS DESDE MONGODB ATLAS
# ============================================================

@st.cache_data(ttl=60)
def cargar_datos_velox_v2():
    """
    Recupera todos los viajes de 'db.viaje', extrae metadatos
    de fecha (usando fecha del documento o timestamp del ObjectId)
    y normaliza identificadores de conductor para seguimiento longitudinal.
    """
    registros = []
    cursor = db.viaje.find({}).sort("_id", 1)

    for indice, doc in enumerate(cursor, start=1):
        oid = doc.get("_id")
        estado = str(doc.get("estado", "pendiente")).lower().strip()
        costo = float(doc.get("costo", 0))
        asientos = int(doc.get("asientos_disponibles", 0))

        # Identificar conductor / usuario si existe en el documento
        conductor_raw = (
            doc.get("conductor_nombre")
            or doc.get("conductor")
            or doc.get("id_conductor")
            or doc.get("conductor_id")
            or doc.get("usuario_id")
        )
        tiene_id_conductor = conductor_raw is not None and str(conductor_raw).strip() != ""
        id_conductor = str(conductor_raw) if tiene_id_conductor else "Sin conductor asignado"

        # Detectar fecha de registro (campo explícito o fecha de creación del ObjectId)
        fecha_doc = doc.get("fecha") or doc.get("fecha_creacion") or doc.get("created_at")
        if isinstance(fecha_doc, datetime):
            fecha_registro = fecha_doc
        elif isinstance(fecha_doc, str):
            try:
                fecha_registro = pd.to_datetime(fecha_doc)
            except Exception:
                fecha_registro = oid.generation_time.replace(tzinfo=None) if hasattr(oid, "generation_time") else datetime.now()
        else:
            fecha_registro = oid.generation_time.replace(tzinfo=None) if hasattr(oid, "generation_time") else datetime.now()

        # Origen y destino
        origen = (
            doc.get("origen")
            or doc.get("punto_encuentro")
            or doc.get("origen_nombre")
            or "Origen registrado"
        )
        destino = doc.get("destino", "Campus UTVT")

        # Viabilidad e índice operativo estimado (0 a 100%)
        es_exitoso = estado in ESTADOS_EXITOSOS
        estatus_viabilidad = (
            "Viaje Consolidado / Exitoso"
            if es_exitoso
            else "Riesgo Operativo / No Consolidado"
        )

        # Índice heurístico de viabilidad económica/ocupación (0-100)
        puntaje_costo = max(0.0, min(100.0, (40.0 - costo) / 30.0 * 100.0))
        puntaje_asientos = min(100.0, (asientos / 4.0) * 100.0)
        indice_viabilidad = round((puntaje_costo * 0.6 + puntaje_asientos * 0.4), 1)

        if costo <= 15:
            segmento_tarifa = "Económico (≤ $15)"
        elif costo <= 30:
            segmento_tarifa = "Estándar ($16 - $30)"
        else:
            segmento_tarifa = "Tarifa Alta (> $30)"

        registros.append({
            "num_viaje": indice,
            "id_viaje": str(oid),
            "id_conductor": id_conductor,
            "tiene_conductor": tiene_id_conductor,
            "fecha_registro": pd.to_datetime(fecha_registro),
            "fecha_dia": pd.to_datetime(fecha_registro).date(),
            "origen": str(origen),
            "destino": str(destino),
            "costo": costo,
            "asientos_disponibles": asientos,
            "estado": estado,
            "estatus_viabilidad": estatus_viabilidad,
            "segmento_tarifa": segmento_tarifa,
            "indice_viabilidad": indice_viabilidad,
            "viaje_exitoso": 1 if es_exitoso else 0
        })

    return pd.DataFrame(registros)


# ============================================================
# 3. INTERFAZ PRINCIPAL v2
# ============================================================

def main():
    st.title("🚗 Velox UTVT — Dashboard Analítico Longitudinal (v2)")
    st.markdown(
        "Seguimiento evolutivo de trayectos, tarifas, disponibilidad de asientos "
        "y desempeño por conductor en **MongoDB Atlas**[cite: 17]."
    )

    if st.sidebar.button("🔄 Sincronizar con MongoDB Atlas"):
        st.cache_data.clear()

    try:
        df = cargar_datos_velox_v2()
    except Exception as error:
        st.error(f"Error de conexión con MongoDB Atlas: {error}")
        st.stop()

    if df.empty:
        st.warning("No se encontraron viajes registrados en 'db.viaje'.")
        st.stop()

    # --------------------------------------------------------
    # BARRA LATERAL DE FILTROS AVANZADOS
    # --------------------------------------------------------
    st.sidebar.header("Filtros de Exploración")

    estados_disp = sorted(df["estado"].unique().tolist())
    estados_sel = st.sidebar.multiselect(
        "Estado del viaje:",
        options=estados_disp,
        default=estados_disp
    )

    segmentos_disp = sorted(df["segmento_tarifa"].unique().tolist())
    segmentos_sel = st.sidebar.multiselect(
        "Segmento tarifario:",
        options=segmentos_disp,
        default=segmentos_disp
    )

    # Filtro por rango de fechas
    fecha_min = df["fecha_dia"].min()
    fecha_max = df["fecha_dia"].max()
    rango_fechas = st.sidebar.date_input(
        "Rango de fechas:",
        value=(fecha_min, fecha_max),
        min_value=fecha_min,
        max_value=fecha_max
    )

    df_filtrado = df[
        (df["estado"].isin(estados_sel)) &
        (df["segmento_tarifa"].isin(segmentos_sel))
    ].copy()

    if isinstance(rango_fechas, tuple) and len(rango_fechas) == 2:
        inicio, fin = rango_fechas
        df_filtrado = df_filtrado[
            (df_filtrado["fecha_dia"] >= inicio) &
            (df_filtrado["fecha_dia"] <= fin)
        ]

    if df_filtrado.empty:
        st.info("No hay registros que coincidan con los filtros seleccionados.")
        st.stop()

    # --------------------------------------------------------
    # KPIs GLOBALES Y DE COBERTURA (ETAPA 10)
    # --------------------------------------------------------
    st.subheader("Indicadores Operativos y Cobertura")

    c1, c2, c3, c4, c5 = st.columns(5)

    total_viajes = len(df_filtrado)
    conductores_unicos = df_filtrado.loc[df_filtrado["tiene_conductor"], "id_conductor"].nunique()
    costo_prom = df_filtrado["costo"].mean()
    asientos_totales = int(df_filtrado["asientos_disponibles"].sum())
    tasa_exito = df_filtrado["viaje_exitoso"].mean() * 100

    c1.metric("Viajes Analizados", f"{total_viajes}")
    c2.metric("Conductores Únicos", f"{conductores_unicos}")
    c3.metric("Tarifa Promedio", f"${costo_prom:.2f} MXN")
    c4.metric("Asientos Totales", f"{asientos_totales}")
    c5.metric("Tasa de Consolidación", f"{tasa_exito:.1f}%")

    st.divider()

    # --------------------------------------------------------
    # SECCION 1: EVOLUCION TEMPORAL DE VIAJES (LONGITUDINAL)
    # --------------------------------------------------------
    st.subheader("📈 Evolución Temporal de Tarifas e Índice de Viabilidad")

    col_evo1, col_evo2 = st.columns(2)

    with col_evo1:
        fig_evo_costo = px.line(
            df_filtrado,
            x="fecha_registro",
            y="costo",
            markers=True,
            text="costo",
            title="Evolución Cronológica de Tarifas ($ MXN)",
            color_discrete_sequence=["#27ae60"],
            labels={"fecha_registro": "Fecha / Registro", "costo": "Costo ($ MXN)"}
        )
        fig_evo_costo.update_traces(textposition="top center")
        st.plotly_chart(fig_evo_costo, width="stretch")

    with col_evo2:
        fig_evo_viab = px.line(
            df_filtrado,
            x="fecha_registro",
            y="indice_viabilidad",
            markers=True,
            title="Evolución del Índice de Viabilidad Operativa (%)",
            color_discrete_sequence=["#2ecc71"],
            labels={"fecha_registro": "Fecha / Registro", "indice_viabilidad": "Índice Viabilidad (%)"}
        )
        fig_evo_viab.update_yaxes(range=[0, 105])
        st.plotly_chart(fig_evo_viab, width="stretch")

    st.divider()

    # --------------------------------------------------------
    # SECCION 2: ANALISIS COMPARATIVO Y SEGMENTACION
    # --------------------------------------------------------
    col_g1, col_g2 = st.columns(2)

    with col_g1:
        st.subheader("Distribución por Estado Operativo")
        conteo_est = df_filtrado["estado"].value_counts().reset_index()
        conteo_est.columns = ["Estado", "Viajes"]

        fig_bar = px.bar(
            conteo_est,
            x="Estado",
            y="Viajes",
            text="Viajes",
            color="Estado",
            color_discrete_sequence=["#27ae60", "#52be80", "#f39c12", "#e74c3c"]
        )
        fig_bar.update_layout(showlegend=False)
        st.plotly_chart(fig_bar, width="stretch")

    with col_g2:
        st.subheader("Costo vs. Asientos Disponibles")
        fig_scatter = px.scatter(
            df_filtrado,
            x="costo",
            y="asientos_disponibles",
            color="estatus_viabilidad",
            size="indice_viabilidad",
            hover_data=["id_viaje", "estado", "id_conductor"],
            color_discrete_map={
                "Viaje Consolidado / Exitoso": "#27ae60",
                "Riesgo Operativo / No Consolidado": "#e74c3c"
            },
            labels={
                "costo": "Costo ($ MXN)",
                "asientos_disponibles": "Asientos Disponibles",
                "estatus_viabilidad": "Estatus"
            }
        )
        st.plotly_chart(fig_scatter, width="stretch")

    st.divider()

    # --------------------------------------------------------
    # SECCION 3: SEGUIMIENTO INDIVIDUAL POR CONDUCTOR / VIAJE
    # --------------------------------------------------------
    st.subheader("🔍 Inspección Longitudinal por Conductor")

    conductores_lista = sorted(df_filtrado["id_conductor"].unique().tolist())
    conductor_elegido = st.selectbox(
        "Selecciona un conductor para revisar su historial:",
        options=conductores_lista
    )

    df_conductor = df_filtrado[df_filtrado["id_conductor"] == conductor_elegido].copy()

    k1, k2, k3 = st.columns(3)
    k1.metric("Viajes del Conductor", len(df_conductor))
    k2.metric("Costo Promedio Ofertado", f"${df_conductor['costo'].mean():.2f} MXN")
    k3.metric("Viabilidad Promedio", f"{df_conductor['indice_viabilidad'].mean():.1f}%")

    columnas_tabla = [
        "num_viaje",
        "fecha_registro",
        "id_viaje",
        "destino",
        "costo",
        "asientos_disponibles",
        "estado",
        "indice_viabilidad"
    ]
    st.dataframe(df_conductor[columnas_tabla], width="stretch")


if __name__ == "__main__":
    main()