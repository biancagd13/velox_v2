# ============================================================
# PROYECTO: VELOX UTVT
# PRACTICA 17 (ETAPA 8): DASHBOARD ANALITICO INTERACTIVO
#
# Archivo:
# p17dashboard_velox.py
#
# Objetivo:
# Visualizar en tiempo real los indicadores clave, estados
# operativos, tarifas y disponibilidad de asientos de los
# viajes registrados en MongoDB Atlas para Velox UTVT.
#
# Ejecución desde terminal:
# streamlit run p17dashboard_velox.py
# ============================================================

import os
import sys
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
    page_title="Velox UTVT - Dashboard Analítico",
    page_icon="🚗",
    layout="wide"
)

ESTADOS_EXITOSOS = ["activo", "aceptado", "en_curso", "finalizado"]


# ============================================================
# 2. CARGA DE DATOS DESDE MONGODB ATLAS
# ============================================================

@st.cache_data(ttl=60)
def cargar_datos_velox():
    """
    Consulta la colección 'viaje' de Velox UTVT en MongoDB Atlas
    y prepara el DataFrame para visualización.
    """
    registros = []
    cursor = db.viaje.find({})

    for doc in cursor:
        estado = str(doc.get("estado", "pendiente")).lower().strip()
        costo = float(doc.get("costo", 0))
        asientos = int(doc.get("asientos_disponibles", 0))

        # Detectar posible campo de origen según estructura del documento
        origen = (
            doc.get("origen")
            or doc.get("punto_encuentro")
            or doc.get("origen_nombre")
            or "Punto registrado"
        )
        destino = doc.get("destino", "Campus UTVT")

        # Clasificar viabilidad operativa
        es_exitoso = estado in ESTADOS_EXITOSOS
        estatus_viabilidad = (
            "Viaje Exitoso / Consolidado"
            if es_exitoso
            else "Riesgo Operativo / Pendiente"
        )

        # Segmento tarifario
        if costo <= 15:
            segmento_tarifa = "Económico (≤ $15)"
        elif costo <= 30:
            segmento_tarifa = "Estándar ($16 - $30)"
        else:
            segmento_tarifa = "Tarifa Alta (> $30)"

        registros.append({
            "id_viaje": str(doc.get("_id")),
            "origen": str(origen),
            "destino": str(destino),
            "costo": costo,
            "asientos_disponibles": asientos,
            "estado": estado,
            "estatus_viabilidad": estatus_viabilidad,
            "segmento_tarifa": segmento_tarifa,
            "viaje_exitoso": 1 if es_exitoso else 0
        })

    return pd.DataFrame(registros)


# ============================================================
# 3. INTERFAZ PRINCIPAL DEL DASHBOARD
# ============================================================

def main():
    st.title("🚗 Velox UTVT — Panel Analítico de Carpooling")
    st.markdown(
        "Monitoreo en tiempo real de viajes universitarios, tarifas, "
        "ocupación y estado operativo desde **MongoDB Atlas**."
    )

    # Botón para recargar datos manualmente
    if st.sidebar.button("🔄 Actualizar datos desde MongoDB"):
        st.cache_data.clear()

    try:
        df = cargar_datos_velox()
    except Exception as error:
        st.error(f"Error al conectar con MongoDB Atlas: {error}")
        st.stop()

    if df.empty:
        st.warning("No hay viajes registrados en la colección 'viaje' de Velox UTVT.")
        st.stop()

    # --------------------------------------------------------
    # BARRA LATERAL DE FILTROS
    # --------------------------------------------------------
    st.sidebar.header("Filtros Operativos")

    estados_disponibles = sorted(df["estado"].unique().tolist())
    estados_seleccionados = st.sidebar.multiselect(
        "Estado del viaje:",
        options=estados_disponibles,
        default=estados_disponibles
    )

    segmentos_disponibles = sorted(df["segmento_tarifa"].unique().tolist())
    segmentos_seleccionados = st.sidebar.multiselect(
        "Segmento tarifario:",
        options=segmentos_disponibles,
        default=segmentos_disponibles
    )

    df_filtrado = df[
        (df["estado"].isin(estados_seleccionados)) &
        (df["segmento_tarifa"].isin(segmentos_seleccionados))
    ].copy()

    if df_filtrado.empty:
        st.info("No hay viajes que coincidan con los filtros seleccionados.")
        st.stop()

    # --------------------------------------------------------
    # INDICADORES CLAVE (KPIs)
    # --------------------------------------------------------
    st.subheader("Indicadores Clave de Operación")

    col1, col2, col3, col4 = st.columns(4)

    total_viajes = len(df_filtrado)
    costo_promedio = df_filtrado["costo"].mean()
    total_asientos = int(df_filtrado["asientos_disponibles"].sum())
    tasa_exito = (df_filtrado["viaje_exitoso"].mean()) * 100

    col1.metric("Total de Viajes", f"{total_viajes}")
    col2.metric("Tarifa Promedio", f"${costo_promedio:.2f} MXN")
    col3.metric("Asientos Ofertados", f"{total_asientos}")
    col4.metric("Tasa de Viabilidad", f"{tasa_exito:.1f}%")

    st.divider()

    # --------------------------------------------------------
    # GRAFICAS INTERACTIVAS
    # --------------------------------------------------------
    graf_col1, graf_col2 = st.columns(2)

    with graf_col1:
        st.subheader("Distribución por Estado Operativo")
        conteo_estados = (
            df_filtrado["estado"]
            .value_counts()
            .reset_index()
        )
        conteo_estados.columns = ["Estado", "Cantidad"]

        fig_estados = px.bar(
            conteo_estados,
            x="Estado",
            y="Cantidad",
            text="Cantidad",
            color="Estado",
            color_discrete_sequence=["#27ae60", "#52be80", "#f39c12", "#e74c3c"]
        )
        fig_estados.update_layout(showlegend=False, xaxis_title="", yaxis_title="Viajes")
        st.plotly_chart(fig_estados, use_container_width=True)

    with graf_col2:
        st.subheader("Relación Costo vs. Asientos Disponibles")
        fig_dispersion = px.scatter(
            df_filtrado,
            x="costo",
            y="asientos_disponibles",
            color="estatus_viabilidad",
            size="asientos_disponibles",
            hover_data=["id_viaje", "estado", "destino"],
            color_discrete_map={
                "Viaje Exitoso / Consolidado": "#27ae60",
                "Riesgo Operativo / Pendiente": "#e74c3c"
            },
            labels={
                "costo": "Costo del Viaje ($ MXN)",
                "asientos_disponibles": "Asientos Disponibles",
                "estatus_viabilidad": "Estatus"
            }
        )
        st.plotly_chart(fig_dispersion, use_container_width=True)

    # Segunda fila de gráficas
    graf_col3, graf_col4 = st.columns(2)

    with graf_col3:
        st.subheader("Proporción de Viabilidad")
        conteo_viabilidad = (
            df_filtrado["estatus_viabilidad"]
            .value_counts()
            .reset_index()
        )
        conteo_viabilidad.columns = ["Estatus", "Total"]

        fig_pie = px.pie(
            conteo_viabilidad,
            names="Estatus",
            values="Total",
            hole=0.4,
            color="Estatus",
            color_discrete_map={
                "Viaje Exitoso / Consolidado": "#27ae60",
                "Riesgo Operativo / Pendiente": "#e74c3c"
            }
        )
        st.plotly_chart(fig_pie, use_container_width=True)

    with graf_col4:
        st.subheader("Tarifa Promedio por Estado")
        tarifa_estado = (
            df_filtrado.groupby("estado", as_index=False)["costo"]
            .mean()
            .round(2)
        )
        fig_tarifa = px.bar(
            tarifa_estado,
            x="estado",
            y="costo",
            text="costo",
            color_discrete_sequence=["#2ecc71"],
            labels={"estado": "Estado", "costo": "Costo Promedio ($ MXN)"}
        )
        st.plotly_chart(fig_tarifa, use_container_width=True)

    st.divider()

    # --------------------------------------------------------
    # TABLA DETALLADA DE VIAJES
    # --------------------------------------------------------
    st.subheader("Detalle de Viajes Registrados")
    columnas_mostrar = [
        "id_viaje",
        "destino",
        "costo",
        "asientos_disponibles",
        "estado",
        "segmento_tarifa",
        "estatus_viabilidad"
    ]
    st.dataframe(df_filtrado[columnas_mostrar], use_container_width=True)


if __name__ == "__main__":
    main()