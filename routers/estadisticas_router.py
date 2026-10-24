# ==========================================================
# MÓDULO DE ESTADÍSTICAS Y GRÁFICAS PROFESIONALES
# Proyecto: Velox UTVT
# ==========================================================

import os

# Configuramos matplotlib con backend sin interfaz gráfica para ejecución segura en FastAPI
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

# Importamos la conexión centralizada a la base de datos de MongoDB
from database import db

# Importamos herramientas de FastAPI para enrutamiento, peticiones HTTP y respuestas
from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

# Inicializamos el enrutador para el módulo de estadísticas con su prefijo institucional y etiqueta
router_estadisticas = APIRouter(prefix="/estadisticas", tags=["Estadísticas y Gráficas"])

# Configuramos el motor de plantillas HTML apuntando a la carpeta 'templates' del proyecto
motor_plantillas = Jinja2Templates(directory="templates")

# Definimos la ruta de la carpeta física donde se guardarán temporalmente las gráficas generadas
CARPETA_STATIC_GRAFICAS = "static/graficas"
os.makedirs(CARPETA_STATIC_GRAFICAS, exist_ok=True)

def generar_grafica_rutas_mas_usadas():
    """Consulta en tiempo real MongoDB, agrupa rutas únicas por origen-destino y genera una gráfica limpia y profesional."""
    try:
        # Consultamos todas las rutas y viajes registrados en la base de datos
        lista_rutas_registradas = list(db["ruta"].find({}, {"_id": 0}))
        lista_viajes = list(db["viaje"].find({}, {"_id": 0}))
        
        # Diccionarios para agrupar sin duplicados
        mapa_rutas = {}
        conteo_por_etiqueta = {}

        # 1. Mapeamos y registramos todas las rutas únicas del catálogo
        for ruta in lista_rutas_registradas:
            id_r = ruta.get("id_ruta")
            origen = ruta.get("origen", "Origen")
            destino = ruta.get("destino", "Destino")
            
            etiqueta_ruta = f"{origen} a {destino}"
            mapa_rutas[id_r] = etiqueta_ruta
            
            # Inicializamos cada ruta única con 0 viajes por defecto
            if etiqueta_ruta not in conteo_por_etiqueta:
                conteo_por_etiqueta[etiqueta_ruta] = 0

        # 2. Contamos y sumamos los viajes reales agrupándolos por su ruta única
        for viaje in lista_viajes:
            id_r = viaje.get("id_ruta")
            if id_r in mapa_rutas:
                etiqueta_ruta = mapa_rutas[id_r]
                conteo_por_etiqueta[etiqueta_ruta] += 1

        # Extraemos las listas limpias para Matplotlib
        nombres_rutas = list(conteo_por_etiqueta.keys())
        conteo_viajes_por_ruta = list(conteo_por_etiqueta.values())

        # Si no hay datos registrados en MongoDB, utilizamos datos de respaldo simulados
        if not nombres_rutas or sum(conteo_viajes_por_ruta) == 0:
            nombres_rutas = ["Toluca - UTVT", "Metepec - UTVT", "Lerma - UTVT", "Almoloya - UTVT"]
            conteo_viajes_por_ruta = [12, 25, 8, 15]

        # ==========================================================
        # CONFIGURACIÓN GENERAL: TAMAÑO DE LA FIGURA AMPLIADO
        # ==========================================================
        plt.figure(figsize=(14, 7))

        # ==========================================================
        # GRÁFICA PRINCIPAL: DIBUJA LA LÍNEA CON EL VERDE INSTITUCIONAL
        # ==========================================================
        plt.plot(
            nombres_rutas,
            conteo_viajes_por_ruta,
            color="#018701",  # Verde institucional Velox UTVT
            linewidth=3,
            marker="o",
            markersize=8,
            markerfacecolor="white",
            markeredgewidth=2,
            label="Viajes programados"
        )

        # ==========================================================
        # TÍTULO PRINCIPAL Y SUBTÍTULO
        # ==========================================================
        plt.suptitle(
            "Análisis de Demanda: Rutas Más Usadas en Velox UTVT",
            fontsize=16,
            fontweight="bold",
            y=0.98
        )

        plt.title(
            "Periodo Activo - Monitoreo en Tiempo Real",
            fontsize=11,
            color="gray",
            pad=15
        )

        # ==========================================================
        # ETIQUETAS EN LOS EJES Y LÍMITES DINÁMICOS
        # ==========================================================
        plt.xlabel(
            "Rutas (Origen - Destino)",
            fontsize=13,
            fontweight="bold",
            labelpad=12
        )

        plt.ylabel(
            "Número de Viajes Registrados",
            fontsize=13,
            fontweight="bold"
        )

        # Ajustamos el límite vertical
        valor_maximo_datos = max(conteo_viajes_por_ruta) if conteo_viajes_por_ruta else 10
        plt.ylim(-0.5, max(6, valor_maximo_datos + 4))

        # ==========================================================
        # ROTACIÓN Y ALINEACIÓN DE ETIQUETAS EN EL EJE X
        # ==========================================================
        plt.xticks(rotation=15, ha="right", fontsize=10, fontweight="semibold")

        # ==========================================================
        # CUADRÍCULA PARA FACILITAR LA LECTURA
        # ==========================================================
        plt.grid(
            linestyle="--",
            linewidth=0.7,
            alpha=0.6
        )

        # ==========================================================
        # LEYENDA
        # ==========================================================
        plt.legend(
            fontsize=11,
            loc="upper left"
        )

        # ==========================================================
        # ANOTACIONES: VALOR EXACTO DE CADA OBSERVACIÓN
        # ==========================================================
        for posicion_x, valor_y in zip(nombres_rutas, conteo_viajes_por_ruta):
            offset_y = 0.35 if valor_y > 0 else 0.2
            plt.text(
                posicion_x,
                valor_y + offset_y,
                str(valor_y),
                fontsize=10,
                ha="center",
                fontweight="semibold"
            )

        # ==========================================================
        # LÍNEA DE PROMEDIO GENERAL
        # ==========================================================
        promedio_general_viajes = np.mean(conteo_viajes_por_ruta) if conteo_viajes_por_ruta else 0

        plt.axhline(
            promedio_general_viajes,
            color="red",
            linestyle="--",
            linewidth=2,
            label="Promedio de Demanda"
        )

        posicion_x_texto = max(0, len(nombres_rutas) - 1.8)
        plt.text(
            posicion_x_texto,
            promedio_general_viajes + 0.35,
            f"Promedio = {promedio_general_viajes:.1f} viajes",
            color="red",
            fontsize=10,
            fontweight="bold",
            bbox=dict(facecolor='white', alpha=0.85, edgecolor='none', pad=2)
        )

        # ==========================================================
        # DESTACAR RUTA MÁS POPULAR (Flecha azul con offset para no amontonar)
        # ==========================================================
        if conteo_viajes_por_ruta:
            indice_ruta_maxima = np.argmax(conteo_viajes_por_ruta)
            nombre_ruta_popular = nombres_rutas[indice_ruta_maxima]
            valor_ruta_popular = conteo_viajes_por_ruta[indice_ruta_maxima]

            # Punto verde oscuro sobre la marcación más alta
            plt.scatter(
                nombre_ruta_popular,
                valor_ruta_popular,
                color="#005200",
                s=180,
                zorder=5,
                label="Ruta Más Popular"
            )

            # Flecha AZUL que apunta un poco arriba del punto para no amontonarse
            plt.annotate(
                "Mayor demanda",
                xy=(indice_ruta_maxima, valor_ruta_popular + 0.45),  # Punta de la flecha (separada del punto verde)
                xytext=(indice_ruta_maxima + 0.15, valor_ruta_popular + 1.8),  # Posición del texto "Mayor demanda"
                arrowprops=dict(
                    arrowstyle="->",
                    color="#0D6EFD",  # Color Azul
                    lw=2.5,
                    mutation_scale=15
                ),
                fontsize=10,
                fontweight="bold",
                color="#0D6EFD"  # Texto también en azul para coordinar estilo
            )

        # ==========================================================
        # PIE DE FIGURA CON FUENTE Y AUTORÍA INSTITUCIONAL
        # ==========================================================
        plt.figtext(
            0.01,
            0.01,
            "Conclusión: Análisis de movilidad | Elaboración propia | Velox UTVT",
            fontsize=9,
            color="gray"
        )

        # ==========================================================
        # QUITAR MARCO SUPERIOR Y DERECHO
        # ==========================================================
        ejes_grafica = plt.gca()
        ejes_grafica.spines["top"].set_visible(False)
        ejes_grafica.spines["right"].set_visible(False)

        # ==========================================================
        # AJUSTAR ESPACIOS Y GUARDAR IMAGEN
        # ==========================================================
        plt.subplots_adjust(top=0.85, bottom=0.25)
        
        ruta_archivo_imagen = os.path.join(CARPETA_STATIC_GRAFICAS, "rutas_mas_usadas.png")
        plt.savefig(
            ruta_archivo_imagen,
            dpi=300,
            bbox_inches="tight"
        )
        plt.close() # Cerramos la figura para liberar memoria RAM del servidor

        return True
    except Exception as error_generacion:
        print(f"Error al generar la gráfica estadística: {error_generacion}")
        plt.close()
        return False

@router_estadisticas.get("/", response_class=HTMLResponse)
def vista_estadisticas(peticion_http: Request):
    """Ruta HTTP GET para generar la gráfica estadística en tiempo real y renderizarla en la interfaz web."""
    generar_grafica_rutas_mas_usadas()
    
    return motor_plantillas.TemplateResponse(
        peticion_http,
        "estadisticas/estadisticas_inicio.html",
        {"imagen_grafica": "/static/graficas/rutas_mas_usadas.png"}
    )