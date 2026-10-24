# ==========================================================
# IMPORTACIÓN DE LIBRERÍAS Y MÓDULOS DEL SISTEMA
# ==========================================================
import os
import matplotlib
# Configuramos el backend de Matplotlib en modo 'Agg' para ejecución segura en servidores web FastAPI
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

# Importación de la conexión centralizada a la base de datos de MongoDB
from database import db

# Importación de herramientas del framework FastAPI para enrutamiento y respuestas web
from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

# Inicialización y configuración del enrutador para el módulo de estadísticas de calificaciones
router_calificaciones_estadisticas = APIRouter(prefix="/estadisticas-calificaciones", tags=["Estadísticas de Calificaciones"])

# Configuración del motor de plantillas HTML apuntando a la carpeta 'templates' del proyecto
motor_plantillas = Jinja2Templates(directory="templates")

# Definición de la ruta física donde se guardarán temporalmente las gráficas de calificaciones
CARPETA_STATIC_GRAFICAS = "static/graficas"
os.makedirs(CARPETA_STATIC_GRAFICAS, exist_ok=True)

def generar_grafica_calificaciones_institucional():
    """
    Función que consulta la base de datos MongoDB para agrupar las calificaciones de 1 a 5 estrellas,
    aplicando una paleta de tonos de verde en degradado ascendente (de claro a verde institucional fuerte).
    """
    try:
        # Consultamos todas las calificaciones registradas en la base de datos MongoDB
        lista_calificaciones_bd = list(db["calificacion"].find({}, {"_id": 0, "puntuacion": 1}))
        
        # Inicializamos un diccionario para contar cuántas calificaciones hay de cada nivel (1 a 5 estrellas)
        conteo_estrellas = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0}

        # Recorremos cada calificación obtenida de la base de datos y sumamos al nivel correspondiente
        for calificacion in lista_calificaciones_bd:
            puntuacion = int(calificacion.get("puntuacion", 5))
            if puntuacion in conteo_estrellas:
                conteo_estrellas[puntuacion] += 1

        # Definimos las categorías fijas para el eje X (De 1 a 5 estrellas)
        categorias_estrellas = ["1 Estrella", "2 Estrellas", "3 Estrellas", "4 Estrellas", "5 Estrellas"]
        
        # Extraemos los totales contados para cada estrella
        totales_conteo = [
            conteo_estrellas[1],
            conteo_estrellas[2],
            conteo_estrellas[3],
            conteo_estrellas[4],
            conteo_estrellas[5]
        ]

        # Si no hay suficientes registros en la base de datos, cargamos datos de respaldo lógicos para demostración
        if sum(totales_conteo) == 0:
            totales_conteo = [2, 4, 8, 15, 35]

        # ==========================================================
        # PALETA DE COLORES EN DEGRADADO DE VERDE (1 a 5 Estrellas)
        # ==========================================================
        colores_degradados_verde = [
            "#C8E6C9",  # 1 Estrella: Verde muy claro (Pastel/Suave)
            "#72C675",  # 2 Estrellas: Verde claro medio
            "#22D028E3",  # 3 Estrellas: Verde estándar
            "#128616D1",  # 4 Estrellas: Verde medio oscuro
            "#085B0B"   # 5 Estrellas: Verde institucional Velox UTVT (Fuerte)
        ]

        # ==========================================================
        # CONFIGURACIÓN GENERAL DEL TAMAÑO DE LA FIGURA
        # ==========================================================
        plt.figure(figsize=(8, 4.5))

        # ==========================================================
        # GRÁFICA PRINCIPAL DE BARRAS CON DEGRADADO
        # ==========================================================
        barras = plt.bar(
            categorias_estrellas,
            totales_conteo,
            color=colores_degradados_verde,  # Lista de colores asignados por barra
            edgecolor="#005200",
            linewidth=1.2,
            width=0.55,
            label="Total"
        )

        # ==========================================================
        # TÍTULO PRINCIPAL Y SUBTÍTULO
        # ==========================================================
        plt.suptitle(
            "Satisfacción del Servicio Velox UTVT",
            fontsize=18,
            fontweight="bold",
            y=0.98
        )

        plt.title(
            "Evaluaciones de Usuarios (1 a 5 Estrellas)",
            fontsize=11,
            color="gray",
            pad=15
        )

        # ==========================================================
        # ETIQUETAS EN LOS EJES Y LÍMITES
        # ==========================================================
        plt.xlabel(
            "Nivel de Puntuación",
            fontsize=12,
            fontweight="bold",
            labelpad=10
        )

        plt.ylabel(
            "Número de Evaluaciones",
            fontsize=12,
            fontweight="bold"
        )

        # Ajuste dinámico del límite del eje Y basado en los datos
        valor_maximo_datos = max(totales_conteo) if totales_conteo else 10
        plt.ylim(0, valor_maximo_datos + 8)

        # ==========================================================
        # CUADRÍCULA PARA FACILITAR LA LECTURA
        # ==========================================================
        plt.grid(
            linestyle="--",
            linewidth=0.7,
            alpha=0.6,
            axis="y"
        )

        # ==========================================================
        # LEYENDA
        # ==========================================================
        plt.legend(
            fontsize=10,
            loc="upper left"
        )

        # ==========================================================
        # ANOTACIONES: MUESTRAN EL VALOR EXACTO SOBRE CADA BARRA
        # ==========================================================
        for barra in barras:
            altura_barra = barra.get_height()
            plt.text(
                barra.get_x() + barra.get_width() / 2.,
                altura_barra + 0.6,
                str(int(altura_barra)),
                ha="center",
                va="bottom",
                fontsize=11,
                fontweight="bold"
            )

        # ==========================================================
        # PIE DE FIGURA INSTITUCIONAL
        # ==========================================================
        plt.figtext(
            0.01,
            0.01,
            "Conclusión: Análisis de satisfacción | Elaboración propia | Velox UTVT",
            fontsize=9,
            color="gray"
        )

        # ==========================================================
        # QUITAR MARCO SUPERIOR Y DERECHO
        # ==========================================================
        ejes_actuales = plt.gca()
        ejes_actuales.spines["top"].set_visible(False)
        ejes_actuales.spines["right"].set_visible(False)

        # ==========================================================
        # AJUSTAR ESPACIOS Y GUARDAR IMAGEN
        # ==========================================================
        plt.subplots_adjust(top=0.85, bottom=0.20)

        ruta_archivo_calificaciones = os.path.join(CARPETA_STATIC_GRAFICAS, "calificaciones_desempeno.png")
        plt.savefig(
            ruta_archivo_calificaciones,
            dpi=300,
            bbox_inches="tight"
        )
        plt.close()  # Liberamos la memoria de la figura

        return True
    except Exception as error_ejecucion:
        print(f"Error al generar la gráfica de calificaciones: {error_ejecucion}")
        plt.close()
        return False

@router_calificaciones_estadisticas.get("/", response_class=HTMLResponse)
def vista_estadisticas_calificaciones(peticion_http: Request):
    """Ruta HTTP GET para generar la gráfica institucional de calificaciones y renderizarla en la web."""
    generar_grafica_calificaciones_institucional()
    
    return motor_plantillas.TemplateResponse(
        peticion_http,
        "estadisticas/calificaciones_estadisticas_inicio.html",
        {"imagen_grafica": "/static/graficas/calificaciones_desempeno.png"}
    )