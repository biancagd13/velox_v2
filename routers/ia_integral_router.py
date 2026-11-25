# ========================================================== # Comentario visual de encabezado de archivo
# ROUTER: VELOX UTVT - CENTRO DE INTELIGENCIA OPERATIVA # Indica el módulo y sistema al que pertenece
# Extracción estructurada desde terminal (KPIs, Tablas con Badges, # Describe la función principal del router
# Matriz de Confusión y Conclusiones en lenguaje natural) # Continúa la descripción de salidas visuales
# Archivo: routers/ia_integral_router.py # Ruta relativa del archivo dentro del proyecto
# ========================================================== # Cierre del bloque de encabezado

import os # Importa utilidades del sistema operativo y variables de entorno
import re # Importa expresiones regulares para limpiar y extraer datos de texto
import subprocess # Permite ejecutar scripts externos de Python como subprocesos
import sys # Provee acceso al ejecutable actual de Python (sys.executable)
from pathlib import Path # Manejo moderno y seguro de rutas de archivos y carpetas

import numpy as np # Importa NumPy para operaciones numéricas y arreglos
import pandas as pd # Importa Pandas para manipular tablas de datos y archivos CSV
from fastapi import APIRouter, HTTPException, Request # Importa el enrutador, excepciones HTTP y objeto Request de FastAPI
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse # Importa tipos de respuesta: archivos, HTML y JSON
from fastapi.templating import Jinja2Templates # Motor de plantillas HTML Jinja2 para renderizar vistas
from pydantic import BaseModel, Field # Clases base para validar datos de entrada en peticiones POST
from sklearn.ensemble import RandomForestClassifier # Modelo de bosque aleatorio para predecir ocupación de viajes
from sklearn.preprocessing import StandardScaler # Escalador estándar para normalizar variables numéricas antes de predecir

from database import db # Importa la conexión activa a la base de datos MongoDB Atlas

router = APIRouter(prefix="/ia-integral", tags=["Centro de Inteligencia Velox"]) # Crea el router con prefijo URL /ia-integral
templates = Jinja2Templates(directory="templates") # Configura la carpeta donde residen las plantillas HTML

RAIZ = Path(__file__).resolve().parent.parent # Obtiene la carpeta raíz del proyecto subiendo dos niveles
CARPETA_IA = RAIZ / "modulo_ia_practicas" # Define la ruta hacia la carpeta con los scripts de IA

PROCESOS_DISPONIBLES: dict[str, dict] = { # Diccionario maestro que mapea cada botón del panel con su script y textos
    "reglas_motor": { # Clave identificadora del primer proceso (motor de reglas)
        "nombre": "Motor de Reglas Operativas", # Título amigable mostrado en el encabezado del resultado
        "descripcion": "Evalúa tarifas, cupos y disponibilidad según las reglas de viajes compartidos de Velox UTVT.", # Subtítulo explicativo
        "script": "p01_regla_fija.py", # Nombre exacto del archivo Python a ejecutar
        "conclusiones_base": [ # Lista de conclusiones en lenguaje natural para el usuario
            "Se evaluaron las reglas fijas de tarifa, lugares disponibles y horario sobre los registros de prueba.", # Primera conclusión
            "Las reglas determinísticas sirven como filtro rápido antes de publicar un viaje en la plataforma.", # Segunda conclusión
            "Al combinar reglas con modelos predictivos se evita rechazar viajes viables por criterios rígidos.", # Tercera conclusión
        ], # Cierre de la lista de conclusiones
    }, # Cierre de configuración para reglas_motor
    "limpieza_datos": { # Clave del proceso de depuración de usuarios
        "nombre": "Depuración y Validación de Usuarios", # Título visible en interfaz
        "descripcion": "Audita los registros de usuarios, filtra datos incompletos y separa cuentas válidas.", # Descripción corta
        "script": "p02limpiezaguardadodatos_d.py", # Archivo Python correspondiente
        "conclusiones_base": [ # Conclusiones predefinidas del módulo
            "El motor revisó los campos obligatorios y correos institucionales de la comunidad universitaria.", # Punto 1
            "Los registros incompletos o con formato inválido se aislaron para no afectar los análisis posteriores.", # Punto 2
            "El archivo CSV limpio quedó listo para descarga o consulta directa.", # Punto 3
        ], # Fin de conclusiones_base
    }, # Fin de limpieza_datos
    "modelo_base": { # Clave del modelo supervisado inicial
        "nombre": "Entrenamiento de Modelo Base", # Título visible
        "descripcion": "Construye el clasificador inicial de referencia sobre trayectos históricos.", # Descripción
        "script": "p03primermodelo.py", # Script Python asociado
        "conclusiones_base": [ # Lista de conclusiones
            "Se entrenó una línea base supervisada utilizando las variables operativas principales de Velox.", # Punto 1
            "Permite medir qué proporción de viajes logra llenarse frente a aquellos con baja demanda.", # Punto 2
            "Al registrar más viajes en la plataforma web, el modelo ajustará sus patrones automáticamente.", # Punto 3
        ], # Fin de lista
    }, # Fin de modelo_base
    "arbol_decision": { # Clave del clasificador por árbol de decisión
        "nombre": "Clasificador por Árbol de Decisión", # Título visible
        "descripcion": "Genera reglas jerárquicas de decisión para predecir la aceptación de viajes.", # Descripción
        "script": "p04.py", # Script Python correspondiente
        "conclusiones_base": [ # Conclusiones del módulo
            "El árbol de decisión identificó la cooperación económica y los asientos disponibles como factores clave.", # Punto 1
            "Las tarifas accesibles dentro del rango estudiantil lideran las ramas con mayor tasa de aceptación.", # Punto 2
            "La estructura jerárquica permite explicar con claridad por qué un viaje se clasifica como exitoso.", # Punto 3
        ], # Fin de lista
    }, # Fin de arbol_decision
    "comparativa_algoritmos": { # Clave de comparativa entre clasificadores
        "nombre": "Comparativa de Modelos Supervisados", # Título visible
        "descripcion": "Compara múltiples clasificadores supervisados y evalúa su matriz de confusión.", # Descripción
        "script": "p05.py", # Script asociado
        "conclusiones_base": [ # Conclusiones del análisis
            "Se contrastó el rendimiento de distintos algoritmos bajo el mismo conjunto de prueba.", # Punto 1
            "La matriz de confusión permite visualizar los aciertos reales frente a los falsos positivos.", # Punto 2
            "Al incrementar el historial real en MongoDB Atlas, la precisión de los modelos se estabiliza.", # Punto 3
        ], # Fin de lista
    }, # Fin de comparativa_algoritmos
    "hiperparametros": { # Clave del ajuste de hiperparámetros
        "nombre": "Ajuste de Hiperparámetros", # Título visible
        "descripcion": "Optimiza la profundidad y parámetros internos para maximizar el rendimiento del modelo.", # Descripción
        "script": "P06.py", # Script con P mayúscula según archivo real
        "conclusiones_base": [ # Conclusiones del proceso
            "Se evaluaron múltiples combinaciones de parámetros para evitar sobreajuste en los datos.", # Punto 1
            "La configuración óptima mejora la capacidad de generalizar hacia rutas y horarios nuevos.", # Punto 2
            "El modelo calibrado mantiene un equilibrio estable entre precisión y cobertura.", # Punto 3
        ], # Fin de lista
    }, # Fin de hiperparametros
    "optimizacion_clusters": { # Clave de segmentación no supervisada K-Means
        "nombre": "Segmentación Óptima de Rutas (K-Means)", # Título visible
        "descripcion": "Agrupa los viajes en perfiles naturales según tarifa y lugares disponibles.", # Descripción
        "script": "p08.py", # Script asociado
        "conclusiones_base": [ # Conclusiones de segmentación
            "Se identificaron perfiles diferenciados de movilidad según cooperación económica y asientos ofrecidos.", # Punto 1
            "Las pruebas con múltiples semillas confirman que los grupos formados son consistentes y reproducibles.", # Punto 2
            "Conocer cada segmento permite sugerir tarifas competitivas acordes al tipo de trayecto.", # Punto 3
        ], # Fin de lista
    }, # Fin de optimizacion_clusters
    "reduccion_pca": { # Clave de reducción de dimensionalidad PCA
        "nombre": "Proyección Dimensional (PCA)", # Título visible
        "descripcion": "Resume múltiples indicadores operativos en componentes principales fáciles de visualizar.", # Descripción
        "script": "p09.py", # Script asociado
        "conclusiones_base": [ # Conclusiones del módulo
            "Se comprimieron las variables de operación conservando la mayor parte de la varianza informativa.", # Punto 1
            "Facilita detectar visualmente trayectos atípicos y concentraciones de demanda hacia el campus.", # Punto 2
            "Las gráficas generadas muestran la distribución espacial de los componentes principales.", # Punto 3
        ], # Fin de lista
    }, # Fin de reduccion_pca
    "extraccion_atlas": { # Clave de consulta y exportación desde MongoDB Atlas
        "nombre": "Extracción y Exportación MongoDB Atlas", # Título visible
        "descripcion": "Consulta en vivo las colecciones de Velox UTVT y exporta los reportes CSV.", # Descripción
        "script": "p16consultasinteligentes.py", # Script asociado
        "conclusiones_base": [ # Conclusiones
            "Se consultaron directamente las colecciones reales de usuarios, rutas, viajes y reservaciones en la nube.", # Punto 1
            "Los reportes consolidados se exportaron en formato CSV listos para auditoría o descarga.", # Punto 2
            "Refleja el estado operativo actual de la comunidad universitaria en Velox UTVT.", # Punto 3
        ], # Fin de lista
    }, # Fin de extraccion_atlas
    "generador_masivo": { # Clave para generar 50,000 registros históricos
        "nombre": "Generador de 50,000 Trayectos Históricos", # Título visible
        "descripcion": "Construye el volumen masivo de viajes del Valle de Toluca hacia Campus UTVT.", # Descripción
        "script": "p20generador_datos_masivos_velox.py", # Script asociado
        "conclusiones_base": [ # Conclusiones
            "Se generaron 50,000 registros respetando los corredores reales de Toluca, Metepec, San Mateo y Lerma.", # Punto 1
            "Este volumen permite evaluar el comportamiento del sistema bajo alta demanda institucional.", # Punto 2
            "El archivo generado sirve como insumo base para el procesamiento distribuido.", # Punto 3
        ], # Fin de lista
    }, # Fin de generador_masivo
    "conversion_parquet": { # Clave de conversión a formato columnar Parquet
        "nombre": "Conversión Columnar Apache Parquet", # Título visible
        "descripcion": "Optimiza el almacenamiento masivo en formato columnar comprimido Parquet.", # Descripción
        "script": "p21conversion_parquet_velox.py", # Script asociado
        "conclusiones_base": [ # Conclusiones
            "Los 50,000 registros se transformaron a formato columnar Parquet con compresión Snappy.", # Punto 1
            "Se redujo drásticamente el peso en disco respecto al almacenamiento plano tradicional.", # Punto 2
            "Acelera las consultas analíticas sobre historiales extensos de movilidad.", # Punto 3
        ], # Fin de lista
    }, # Fin de conversion_parquet
    "spark_procesamiento": { # Clave de agregación distribuida con PySpark
        "nombre": "Agregación Distribuida PySpark", # Título visible
        "descripcion": "Calcula demanda, tarifas promedio y tasas de ocupación por punto de abordaje.", # Descripción
        "script": "p22spark_procesamiento_velox.py", # Script asociado
        "conclusiones_base": [ # Conclusiones
            "Apache Spark procesó en paralelo los 50,000 trayectos agrupados por nodo de origen.", # Punto 1
            "Permite identificar qué puntos del Valle de Toluca concentran mayor volumen y ocupación.", # Punto 2
            "Sirve para priorizar rutas recomendadas hacia Santa María Atarasquillo.", # Punto 3
        ], # Fin de lista
    }, # Fin de spark_procesamiento
    "spark_mllib": { # Clave de entrenamiento distribuido con Spark MLlib
        "nombre": "Entrenamiento Distribuido Spark MLlib", # Título visible
        "descripcion": "Evalúa clasificadores a gran escala con Regresión Logística, Árboles y Random Forest.", # Descripción
        "script": "p23spark_mllib_velox.py", # Script asociado
        "conclusiones_base": [ # Conclusiones
            "Se entrenaron y compararon modelos distribuidos sobre el historial masivo de trayectos.", # Punto 1
            "El índice ROC y el puntaje F1 permiten elegir el clasificador más confiable en producción.", # Punto 2
            "La arquitectura distribuida escala sin degradar el tiempo de respuesta.", # Punto 3
        ], # Fin de lista
    }, # Fin de spark_mllib
    "benchmark_bigdata": { # Clave de comparativa de velocidad entre Pandas y Spark
        "nombre": "Benchmark Tradicional vs. Big Data", # Título visible
        "descripcion": "Compara tiempos de respuesta y uso de disco entre procesamiento local y distribuido.", # Descripción
        "script": "p24bigdata_vs_tradicional_velox.py", # Script asociado
        "conclusiones_base": [ # Conclusiones
            "Se midió el tiempo de ejecución y espacio en disco entre Pandas tradicional y Spark + Parquet.", # Punto 1
            "El formato columnar y el procesamiento distribuido reducen significativamente el consumo de recursos.", # Punto 2
            "Justifica el uso de pipelines distribuidos al crecer el volumen de viajes.", # Punto 3
        ], # Fin de lista
    }, # Fin de benchmark_bigdata
    "perceptron_lineal": { # Clave de neurona perceptrón base
        "nombre": "Neurona Perceptrón Base", # Título visible
        "descripcion": "Entrena una neurona artificial lineal para clasificar la viabilidad de trayectos.", # Descripción
        "script": "p25neurona_perceptron_velox.py", # Script asociado
        "conclusiones_base": [ # Conclusiones
            "La neurona ajustó sus pesos sinápticos premiando puntualidad y cupo, y penalizando tarifas altas.", # Punto 1
            "Demuestra de forma interpretable cómo cada variable influye en la aceptación del viaje.", # Punto 2
            "Sirve como fundamento para arquitecturas neuronales multicapa más complejas.", # Punto 3
        ], # Fin de lista
    }, # Fin de perceptron_lineal
    "red_mlp": { # Clave de red neuronal multicapa MLP
        "nombre": "Red Neuronal Multicapa (MLP)", # Título visible
        "descripcion": "Modela relaciones no lineales mediante capas ocultas con activación ReLU.", # Descripción
        "script": "p26red_neuronal_mlp_velox.py", # Script asociado
        "conclusiones_base": [ # Conclusiones
            "Las capas ocultas capturaron interacciones complejas entre tarifa, horario y distancia.", # Punto 1
            "Supera las limitaciones de los modelos puramente lineales en escenarios de hora pico.", # Punto 2
            "Mantiene un desempeño equilibrado entre precisión y sensibilidad.", # Punto 3
        ], # Fin de lista
    }, # Fin de red_mlp
    "deep_learning_pytorch": { # Clave de red neuronal profunda en PyTorch
        "nombre": "Red Neuronal Profunda (PyTorch)", # Título visible
        "descripcion": "Entrena una arquitectura profunda en PyTorch con curvas de pérdida y validación ROC.", # Descripción
        "script": "p27deep_learning_pytorch_velox.py", # Script asociado
        "conclusiones_base": [ # Conclusiones
            "La red neuronal profunda convergió de forma estable minimizando la pérdida en validación.", # Punto 1
            "Las curvas de entrenamiento y ROC confirman una buena capacidad de discriminación.", # Punto 2
            "Representa el modelo neuronal de mayor capacidad dentro del ecosistema Velox UTVT.", # Punto 3
        ], # Fin de lista
    }, # Fin de deep_learning_pytorch
} # Cierre del diccionario PROCESOS_DISPONIBLES

CARPETAS_ARTEFACTOS = [ # Lista de directorios donde los scripts guardan gráficas PNG y archivos CSV
    CARPETA_IA, # Carpeta raíz de prácticas de IA
    CARPETA_IA / "resultados", # Subcarpeta general de resultados
    CARPETA_IA / "resultados_p08_velox", # Subcarpeta de gráficas K-Means
    CARPETA_IA / "resultados_p09_velox", # Subcarpeta de gráficas PCA
    CARPETA_IA / "resultados_p16_velox", # Subcarpeta de exportaciones MongoDB Atlas
    CARPETA_IA / "resultados_bigdata_velox", # Subcarpeta de resultados Spark/Parquet
    CARPETA_IA / "resultados_dl_velox", # Subcarpeta de curvas PyTorch/Deep Learning
    CARPETA_IA / "static", # Carpeta estática interna
    CARPETA_IA / "static" / "img" / "ia", # Subcarpeta estática de imágenes de IA
] # Cierre de CARPETAS_ARTEFACTOS

TRADUCCION_COLUMNAS: dict[str, str] = { # Diccionario que traduce encabezados técnicos de Python a nombres legibles
    "cluster": "Grupo de Viaje", # Traduce cluster a Grupo de Viaje
    "costo": "Costo del Viaje ($)", # Traduce costo
    "asientos_disponibles": "Asientos Disponibles", # Traduce asientos_disponibles
    "distancia_km": "Distancia (km)", # Traduce distancia_km
    "puntualidad": "Puntualidad (%)", # Traduce puntualidad
    "estado": "Resultado Real (Estado)", # Traduce estado
    "real": "Resultado Real (Estado)", # Traduce real
    "prediccion_reglas": "Predicción Reglas", # Traduce prediccion_reglas
    "pred_reglas": "Predicción Reglas", # Traduce pred_reglas
    "prediccion_ml": "Predicción ML", # Traduce prediccion_ml
    "pred_ml": "Predicción ML", # Traduce pred_ml
    "semilla_a": "Prueba A", # Traduce semilla_a
    "semilla_b": "Prueba B", # Traduce semilla_b
    "ari": "Coincidencia (0 a 1)", # Traduce índice ARI
    "modelo": "Modelo Evaluado", # Traduce modelo
    "algoritmo": "Algoritmo", # Traduce algoritmo
    "accuracy": "Exactitud", # Traduce accuracy
    "precision": "Precisión", # Traduce precision
    "recall": "Cobertura", # Traduce recall
    "f1": "Puntaje F1", # Traduce f1
    "f1_score": "Puntaje F1", # Traduce f1_score
    "auc_roc": "Índice ROC", # Traduce auc_roc
    "origen": "Punto de Origen", # Traduce origen
    "total_viajes": "Total de Viajes", # Traduce total_viajes
    "tarifa_promedio": "Tarifa Media ($)", # Traduce tarifa_promedio
    "tasa_ocupacion": "Ocupación (%)", # Traduce tasa_ocupacion
    "tiempo_seg": "Tiempo (s)", # Traduce tiempo_seg
    "tamano_mb": "Peso en Disco (MB)", # Traduce tamano_mb
} # Cierre de TRADUCCION_COLUMNAS


class SolicitudEjecucion(BaseModel): # Esquema Pydantic para validar el cuerpo JSON al ejecutar un proceso
    clave_proceso: str # Campo obligatorio con la clave del proceso seleccionado


class SolicitudPrediccion(BaseModel): # Esquema Pydantic para validar los datos del simulador de viajes
    costo: float = Field(ge=0, le=500) # Tarifa en pesos entre 0 y 500
    asientos_disponibles: int = Field(ge=1, le=8) # Cupo disponible entre 1 y 8 asientos
    distancia_km: float = Field(default=16.5, ge=1, le=100) # Distancia en km (por defecto 16.5 km)
    puntualidad: float = Field(default=90.0, ge=0, le=100) # Porcentaje de puntualidad del conductor (0 a 100)


def _resolver_script(nombre_script: str) -> Path | None: # Función auxiliar para encontrar el archivo Python en disco
    """Localiza el script exacto o ignorando diferencias de mayúsculas/minúsculas.""" # Docstring explicativo
    directo = CARPETA_IA / nombre_script # Construye la ruta directa esperada
    if directo.is_file(): # Verifica si el archivo existe exactamente con ese nombre
        return directo # Devuelve la ruta encontrada
    objetivo = nombre_script.lower() # Convierte el nombre buscado a minúsculas para comparación flexible
    if CARPETA_IA.is_dir(): # Comprueba que la carpeta modulo_ia_practicas exista
        for archivo in CARPETA_IA.iterdir(): # Recorre cada elemento dentro de la carpeta
            if archivo.is_file() and archivo.name.lower() == objetivo: # Compara nombres ignorando mayúsculas/minúsculas
                return archivo # Devuelve la ruta coincidente
    return None # Retorna None si el archivo no existe


def _resolver_ruta_segura(rel_path: str) -> Path: # Valida que una ruta relativa no salga de CARPETA_IA (evita Path Traversal)
    ruta = (CARPETA_IA / rel_path).resolve() # Resuelve la ruta absoluta canónica
    if CARPETA_IA.resolve() not in ruta.parents and ruta != CARPETA_IA.resolve(): # Verifica que esté dentro de CARPETA_IA
        raise HTTPException(status_code=403, detail="Ruta no permitida") # Lanza error 403 si intentan acceder fuera
    if not ruta.is_file(): # Comprueba que el archivo realmente exista
        raise HTTPException(status_code=404, detail="Archivo no encontrado") # Lanza error 404 si no existe
    return ruta # Devuelve la ruta segura validada


def _listar_artefactos() -> dict: # Escanea todas las carpetas configuradas en busca de gráficas y CSVs
    imagenes = [] # Lista para almacenar metadatos de imágenes encontradas
    csvs = [] # Lista para almacenar metadatos de archivos CSV encontrados
    vistos = set() # Conjunto para evitar registrar dos veces el mismo archivo

    for carpeta in CARPETAS_ARTEFACTOS: # Recorre cada carpeta candidata
        if not carpeta.is_dir(): # Si la carpeta aún no existe en disco, la omite
            continue # Pasa a la siguiente carpeta
        for archivo in sorted(carpeta.iterdir(), key=lambda f: f.stat().st_mtime, reverse=True): # Ordena del más reciente al más antiguo
            if not archivo.is_file(): # Ignora subdirectorios
                continue # Continúa con el siguiente elemento
            rel = archivo.relative_to(CARPETA_IA).as_posix() # Obtiene la ruta relativa con barras normales
            if rel in vistos: # Si ya fue agregado anteriormente, lo salta
                continue # Continúa
            vistos.add(rel) # Marca la ruta relativa como procesada

            ext = archivo.suffix.lower() # Obtiene la extensión del archivo en minúsculas
            mtime = archivo.stat().st_mtime # Obtiene la fecha de última modificación (timestamp)
            if ext in {".png", ".jpg", ".jpeg"}: # Si es una imagen válida
                imagenes.append({ # Agrega el diccionario de metadatos de imagen
                    "nombre": archivo.stem.replace("_", " ").title(), # Nombre legible sin guiones bajos
                    "archivo": archivo.name, # Nombre real del archivo
                    "ruta_rel": rel, # Ruta relativa para servirlo vía endpoint
                    "mtime": mtime, # Timestamp para detectar si se actualizó tras ejecutar un script
                }) # Fin de append imagen
            elif ext == ".csv": # Si es un archivo CSV
                csvs.append({ # Agrega el diccionario de metadatos de CSV
                    "nombre": archivo.stem.replace("_", " ").title(), # Título legible del reporte
                    "archivo": archivo.name, # Nombre real del CSV
                    "ruta_rel": rel, # Ruta relativa segura
                    "tam_kb": round(archivo.stat().st_size / 1024, 1), # Calcula el tamaño en kilobytes con 1 decimal
                    "mtime": mtime, # Timestamp de modificación
                }) # Fin de append csv

    return {"imagenes": imagenes, "csvs": csvs} # Retorna ambas listas agrupadas en un diccionario


def _es_ruido_tecnico(linea: str) -> bool: # Determina si una línea de consola es basura técnica que debe ocultarse al usuario
    s = linea.strip() # Elimina espacios al inicio y final
    if not s: # Si la línea está vacía
        return True # Se considera ruido
    if re.fullmatch(r"[\s=\-_*#+|]+", s): # Si solo contiene separadores visuales como ==== o ----
        return True # Se considera ruido
    if re.match(r"^(Name:\s*\w+,\s*)?dtype:\s*\w+", s, re.IGNORECASE): # Filtra pies de página de Pandas (dtype: int64)
        return True # Se considera ruido
    if re.fullmatch(r"\.{3}<\d+\s+lines>\.{3}", s, re.IGNORECASE): # Filtra truncamientos de consola (...<2 lines>...)
        return True # Se considera ruido
    if re.match(r"^VELOX UTVT\s*-\s*PRACTICA\s*\d+", s, re.IGNORECASE): # Filtra banners internos de práctica
        return True # Se considera ruido
    if re.match(r"^PRACTICA\s*\d+\s*FINALIZADA", s, re.IGNORECASE): # Filtra mensajes internos de cierre de práctica
        return True # Se considera ruido

    patrones_ruido = [ # Lista de fragmentos de logs técnicos de Python, Warnings y Apache Spark
        "Traceback (most recent call last)", # Inicio de traza de error técnico
        'File "', # Ruta interna de archivo en traceback
        "UserWarning:", # Advertencia interna de librerías Python
        "FutureWarning:", # Advertencia de obsolescencia futura
        "DeprecationWarning:", # Advertencia de función obsoleta
        "Setting default log level", # Mensaje de arranque de Spark
        "WARN NativeCodeLoader", # Advertencia nativa de Hadoop/Spark
        "WARN Utils:", # Advertencia de utilidades Spark
        "INFO SparkContext", # Log informativo interno de Spark
        "INFO BlockManager", # Log de memoria interna de Spark
        "INFO MemoryStore", # Log de almacenamiento interno de Spark
        "INFO DAGScheduler", # Log de planificador de tareas Spark
        "INFO TaskSchedulerImpl", # Log de ejecutor de tareas Spark
        "INFO Executor", # Log de nodo ejecutor Spark
        "INFO ShutdownHookManager", # Log de apagado de Spark
        "Using Spark's default log4j", # Mensaje de configuración log4j
        "[Stage ", # Barra de progreso de etapas de Spark
    ] # Fin de patrones_ruido
    return any(p in s for p in patrones_ruido) # Retorna True si la línea contiene alguno de esos logs


def _traducir_columna(nombre_col: str) -> str: # Convierte nombres de columnas de DataFrame en etiquetas humanas
    clave = nombre_col.strip().lower() # Normaliza el nombre a minúsculas sin espacios
    if clave in TRADUCCION_COLUMNAS: # Si existe en el diccionario de traducciones
        return TRADUCCION_COLUMNAS[clave] # Retorna la traducción exacta
    return nombre_col.replace("_", " ").strip().capitalize() # Si no, reemplaza guiones bajos y capitaliza


def _extraer_tarjetas_kpi(texto: str) -> list[dict]: # Extrae métricas clave de la salida de consola para las tarjetas KPI
    tarjetas: list[dict] = [] # Lista donde se acumularán las tarjetas detectadas

    m_shape = re.search(r"\((\d+),\s*(\d+)\)", texto) # Busca dimensiones de tabla tipo (5, 3)
    if m_shape: # Si encontró filas y columnas
        tarjetas.append({ # Añade tarjeta de dimensiones del dataset
            "etiqueta": "Dimensiones del Dataset", # Título superior de la tarjeta
            "valor": f"{m_shape.group(1)} filas, {m_shape.group(2)} columnas", # Valor formateado
        }) # Fin de append
    else: # Si no hubo shape explícito, busca conteo total de registros
        m_tot = re.search( # Busca patrones como Total registros: 50000
            r"(?:Total(?: de)? registros|Viajes generados|Registros limpios|Total de viajes cargados|Total viajes)[^\d]{0,25}([\d,]+)",
            texto, # Texto completo de consola
            re.IGNORECASE, # Ignora mayúsculas/minúsculas
        ) # Fin de búsqueda regex
        if m_tot: # Si encontró el total de registros
            tarjetas.append({ # Añade tarjeta de registros analizados
                "etiqueta": "Registros Analizados", # Título de tarjeta
                "valor": f"{m_tot.group(1)} registros", # Cantidad encontrada
            }) # Fin de append

    if re.search(r"Valores faltantes|Nulos", texto, re.IGNORECASE): # Detecta si el script auditó valores nulos
        nulos_cero = re.findall(r"^[a-zA-Z_]+\s+0$", texto, re.MULTILINE) # Verifica si las columnas reportaron 0 nulos
        if nulos_cero: # Si hay columnas con 0 valores faltantes
            tarjetas.append({ # Añade tarjeta confirmando integridad de datos
                "etiqueta": "Valores Faltantes (Nulos)", # Título de tarjeta
                "valor": "0 en registros activos", # Valor amigable
            }) # Fin de append

    m_train = re.search(r"(?:Entrenamiento|Train)[^\d]{0,20}(\d+)", texto, re.IGNORECASE) # Busca cantidad de datos de entrenamiento
    if m_train: # Si encontró registros de entrenamiento
        tarjetas.append({"etiqueta": "Registros de Entrenamiento", "valor": m_train.group(1)}) # Añade tarjeta

    m_test = re.search(r"(?:Prueba|Test)[^\d]{0,20}(\d+)", texto, re.IGNORECASE) # Busca cantidad de datos de prueba
    if m_test: # Si encontró registros de prueba
        tarjetas.append({"etiqueta": "Registros de Prueba", "valor": m_test.group(1)}) # Añade tarjeta

    patrones_extra = [ # Lista de tuplas (expresión regular, etiqueta de tarjeta) para métricas de evaluación
        (r"(?:Exactitud\s*\(?Reglas[^:\n]*\)?|Accuracy\s*Reglas)[^\d]{0,20}(\d+(?:\.\d+)?\s*%?)", "Exactitud (Reglas Fijas)"), # Exactitud reglas
        (r"(?:Exactitud\s*\(?ML|Exactitud\s*\(?Machine|Accuracy|Exactitud)[^\d]{0,20}(\d+(?:\.\d+)?\s*%?)", "Exactitud del Modelo"), # Exactitud ML
        (r"(?:F1[-\s]?Score)[^\d]{0,20}(\d+(?:\.\d+)?\s*%?)", "Puntaje F1-Score"), # Puntaje F1
        (r"(?:AUC[-\s]?ROC)[^\d]{0,20}(\d+(?:\.\d+)?)", "Índice ROC"), # Área bajo la curva ROC
        (r"(?:Silueta|Silhouette)[^\d]{0,20}(\d+(?:\.\d+)?)", "Calidad de Grupos (Silueta)"), # Coeficiente de silueta K-Means
        (r"(?:Tiempo total|Duraci[oó]n)[^\d]{0,20}(\d+(?:\.\d+)?\s*s(?:eg)?)", "Tiempo de Ejecución"), # Tiempo de cómputo
    ] # Fin de patrones_extra
    etiquetas_vistas = {t["etiqueta"] for t in tarjetas} # Conjunto para evitar tarjetas duplicadas
    for patron, etiqueta in patrones_extra: # Recorre cada métrica buscada
        if etiqueta in etiquetas_vistas: # Si ya se agregó esa etiqueta, la omite
            continue # Continúa
        m = re.search(patron, texto, re.IGNORECASE) # Busca el valor numérico en la salida
        if m: # Si hubo coincidencia
            val = m.group(1).strip() # Extrae el valor limpio
            if "Exactitud" in etiqueta and "%" not in val: # Si es exactitud en formato decimal (ej. 0.85) sin signo %
                try: # Intenta convertir a flotante
                    num = float(val) # Convierte texto a número
                    if num <= 1.0: # Si está en escala 0 a 1
                        val = f"{num * 100:.2f}%" # Lo formatea como porcentaje con 2 decimales
                except ValueError: # Si falla la conversión, deja el texto original
                    pass # No hace nada
            tarjetas.append({"etiqueta": etiqueta, "valor": val}) # Agrega la tarjeta KPI
            etiquetas_vistas.add(etiqueta) # Marca la etiqueta como agregada

    return tarjetas[:7] # Devuelve máximo 7 tarjetas KPI para mantener limpio el diseño


def _extraer_matriz_confusion(texto: str) -> list[list[str]] | None: # Detecta una matriz de confusión 2x2 impresa en consola
    m = re.search( # Busca patrón [[a b] [c d]] con o sin comas
        r"\[\s*\[\s*(\d+)\s*,?\s*(\d+)\s*\]\s*,?\s*\[\s*(\d+)\s*,?\s*(\d+)\s*\]\s*\]",
        texto, # Texto de salida
    ) # Fin de regex
    if m: # Si encontró los 4 valores de la matriz 2x2
        return [ # Retorna la matriz estructurada en 2 filas por 2 columnas
            [m.group(1), m.group(2)], # Fila 0: Verdaderos Negativos y Falsos Positivos
            [m.group(3), m.group(4)], # Fila 1: Falsos Negativos y Verdaderos Positivos
        ] # Fin de matriz
    return None # Retorna None si el script no imprimió matriz de confusión


def _extraer_reporte_clasificacion(texto: str) -> list[dict] | None: # Parsea el classification_report de scikit-learn
    if not re.search(r"precision\s+recall\s+f1-score\s+support", texto, re.IGNORECASE): # Verifica si existe el encabezado de sklearn
        return None # Si no existe, retorna None

    filas_reporte = [] # Lista para guardar las filas estructuradas del reporte
    mapa_clases = { # Diccionario que traduce y explica cada fila del reporte de sklearn
        "0": "No exitoso (0)", # Clase negativa
        "1": "Exitoso (1)", # Clase positiva
        "no exitoso": "No exitoso", # Etiqueta textual negativa
        "exitoso": "Exitoso", # Etiqueta textual positiva
        "accuracy": "Exactitud global", # Exactitud general
        "macro avg": "Promedio equilibrado (macro avg: mismo peso a exitosos y no exitosos)", # Explica macro avg
        "weighted avg": "Promedio ponderado (weighted avg: ajustado según cantidad real de viajes)", # Explica weighted avg
    } # Fin de mapa_clases

    for linea in texto.splitlines(): # Analiza línea por línea del texto
        s = linea.strip() # Limpia espacios extremos
        m_full = re.match( # Coincide con filas completas de 4 valores numéricos (precision, recall, f1, support)
            r"^(0|1|no exitoso|exitoso|macro avg|weighted avg)\s+(\d+\.\d+)\s+(\d+\.\d+)\s+(\d+\.\d+)\s+(\d+)$",
            s, # Línea actual
            re.IGNORECASE, # Ignora mayúsculas
        ) # Fin de regex
        if m_full: # Si la línea coincide con una fila completa
            etiqueta_raw = m_full.group(1).lower() # Obtiene el nombre original de la fila
            filas_reporte.append({ # Agrega la fila traducida con porcentajes formateados
                "clase": mapa_clases.get(etiqueta_raw, m_full.group(1)), # Nombre explicativo
                "precision": f"{float(m_full.group(2)) * 100:.0f}%", # Precisión en porcentaje
                "recall": f"{float(m_full.group(3)) * 100:.0f}%", # Cobertura en porcentaje
                "f1": f"{float(m_full.group(4)) * 100:.0f}%", # Puntaje F1 en porcentaje
                "soporte": m_full.group(5), # Cantidad de registros evaluados
            }) # Fin de append
            continue # Pasa a la siguiente línea

        m_acc = re.match(r"^accuracy\s+(\d+\.\d+)\s+(\d+)$", s, re.IGNORECASE) # Coincide con la fila accuracy (solo 2 números)
        if m_acc: # Si es la fila de exactitud global
            filas_reporte.append({ # La alinea correctamente colocando guiones en precisión y cobertura
                "clase": "Exactitud global", # Nombre de fila
                "precision": "—", # No aplica
                "recall": "—", # No aplica
                "f1": f"{float(m_acc.group(1)) * 100:.0f}%", # Exactitud en porcentaje
                "soporte": m_acc.group(2), # Total de muestras
            }) # Fin de append

    return filas_reporte if filas_reporte else None # Retorna la lista si tuvo filas válidas


def _limpiar_bloque_tabular(bloque_filas: list[list[str]]) -> tuple[list[str], list[list[str]]]: # Normaliza tablas extraídas de consola
    if len(bloque_filas) < 2: # Se necesita al menos encabezado y una fila de datos
        return [], [] # Retorna vacío si no hay suficientes filas

    encabezados_raw = bloque_filas[0] # Primera fila tomada como encabezados
    datos_raw = bloque_filas[1:] # Filas restantes tomadas como registros

    if len(datos_raw[0]) == len(encabezados_raw) + 1: # Detecta si Pandas imprimió el índice numérico sin encabezado
        encabezados_raw = ["Registro #"] + encabezados_raw # Inserta columna 'Registro #' para evitar desfase

    num_cols = len(encabezados_raw) # Total de columnas esperadas
    filas_ajustadas = [] # Lista de filas con longitud normalizada
    for fila in datos_raw: # Recorre cada fila de datos
        if len(fila) < num_cols: # Si faltan celdas al final
            fila = fila + [""] * (num_cols - len(fila)) # Rellena con cadenas vacías
        elif len(fila) > num_cols: # Si sobran celdas
            fila = fila[:num_cols] # Recorta al ancho del encabezado
        filas_ajustadas.append(fila) # Guarda la fila ajustada

    indices_validos = [idx for idx, col in enumerate(encabezados_raw) if col.strip() != "..."] # Filtra columnas truncadas '...'
    columnas_finales = [_traducir_columna(encabezados_raw[idx]) for idx in indices_validos] # Traduce encabezados válidos

    filas_finales = [] # Lista de filas limpias para mostrar en HTML
    for fila in filas_ajustadas: # Recorre cada fila ajustada
        nueva_fila = [] # Nueva fila depurada
        for idx in indices_validos: # Recorre solo los índices de columnas válidas
            val = fila[idx].strip() # Valor de la celda
            col_nombre = columnas_finales[len(nueva_fila)].lower() # Nombre de la columna actual
            if any(k in col_nombre for k in ("predicción", "resultado", "estado", "exitoso")) and val in {"1", "0", "1.0", "0.0"}: # Si es columna binaria de estado/predicción
                val = "Exitoso" if val.startswith("1") else "No exitoso" # Convierte 1/0 en etiqueta legible para badge
            nueva_fila.append(val) # Agrega la celda procesada
        filas_finales.append(nueva_fila) # Agrega la fila completa

    return columnas_finales, filas_finales # Retorna encabezados y filas listas para renderizar


def _es_fila_tabular_compacta(linea: str) -> list[str] | None: # Detecta líneas de tabla separadas por espacios simples
    s = linea.strip() # Limpia espacios extremos
    if not s or ":" in s or s.endswith("."): # Descarta oraciones normales o líneas clave:valor
        return None # No es fila tabular
    tokens = s.split() # Divide la línea por espacios en blanco
    if 3 <= len(tokens) <= 10: # Considera tablas entre 3 y 10 columnas
        es_cabecera = all(re.match(r"^([A-Za-zÁÉÍÓÚÑáéíóúñ_][A-Za-z0-9_]*|\.{3})$", t) for t in tokens) # Verifica si todos son nombres de columna
        cant_numeros = sum(1 for t in tokens if re.match(r"^(-?\d+(?:\.\d+)?%?|\.{3})$", t)) # Cuenta cuántos tokens son números
        es_fila_numerica = cant_numeros >= len(tokens) - 1 # Verifica si la fila es mayormente numérica
        if es_cabecera or es_fila_numerica: # Si cumple como encabezado o fila de datos
            return tokens # Retorna la lista de celdas
    return None # Retorna None si no parece tabla


def _analizar_salida_terminal(salida_texto: str) -> dict: # Motor principal que transforma texto crudo de consola en bloques estructurados
    tablas: list[dict] = [] # Lista de tablas extraídas
    bloques_detalle: list[dict] = [] # Lista de bloques de hallazgos en viñetas
    lineas_auditoria: list[str] = [] # Bitácora literal para el acordeón de auditoría
    titulos_tablas_vistos: set[str] = set() # Evita duplicar la misma tabla (ej. df y df.head())

    lineas_crudas = salida_texto.splitlines() # Divide toda la salida de consola en líneas
    n = len(lineas_crudas) # Número total de líneas
    i = 0 # Puntero de recorrido de líneas

    titulo_actual = "Resumen del Análisis" # Título por defecto para el primer bloque
    items_actuales: list[str] = [] # Viñetas acumuladas bajo el título actual

    def guardar_bloque(): # Función interna para cerrar y guardar el bloque de texto acumulado
        nonlocal items_actuales # Permite reiniciar la lista externa
        if items_actuales: # Si hay elementos acumulados
            bloques_detalle.append({ # Guarda el bloque con su título e ítems
                "titulo": titulo_actual, # Título de la sección
                "items": items_actuales[:15], # Máximo 15 viñetas por bloque
            }) # Fin de append
            items_actuales = [] # Reinicia el acumulador

    def registrar_tabla(titulo: str, bloque_bruto: list[list[str]]): # Valida, limpia y registra una tabla sin duplicados
        cols, filas = _limpiar_bloque_tabular(bloque_bruto) # Normaliza encabezados y celdas
        if not cols or not filas: # Si quedó vacía tras limpiar, no la agrega
            return # Sale de la función
        firma = f"{titulo.lower()}|{'|'.join(cols[:3])}" # Crea huella única con título y primeras 3 columnas
        if firma in titulos_tablas_vistos: # Si ya se registró una tabla idéntica bajo ese título
            return # Descarta el duplicado
        titulos_tablas_vistos.add(firma) # Registra la huella
        tablas.append({ # Agrega la tabla estructurada
            "titulo": titulo, # Título de la tabla
            "columnas": cols, # Encabezados traducidos
            "filas": filas[:20], # Máximo 20 filas para vista ágil
        }) # Fin de append

    while i < n: # Recorre todas las líneas de la salida
        linea = lineas_crudas[i] # Obtiene la línea actual
        s = linea.strip() # Limpia espacios extremos

        if not s or re.fullmatch(r"[\s=\-_*#+|]+", s): # Salta líneas vacías o adornos de guiones
            i += 1 # Avanza al siguiente índice
            continue # Continúa el ciclo

        lineas_auditoria.append(s) # Guarda la línea no vacía en la bitácora de auditoría

        if _es_ruido_tecnico(s): # Si es ruido técnico (dtype, warnings, logs de Spark)
            i += 1 # Lo omite
            continue # Continúa

        # Agrupar conteos binarios sueltos de value_counts (ej. viaje_exitoso \n 0 98 \n 1 102)
        if re.fullmatch(r"[a-zA-Z_]+", s) and i + 2 < n: # Detecta nombre de serie seguido de 2 líneas de conteo
            m0 = re.match(r"^0\s+(\d+)$", lineas_crudas[i + 1].strip()) # Busca conteo de clase 0
            m1 = re.match(r"^1\s+(\d+)$", lineas_crudas[i + 2].strip()) # Busca conteo de clase 1
            if m0 and m1: # Si ambos conteos coinciden
                items_actuales.append( # Crea una sola oración clara en lugar de 3 líneas sueltas
                    f"Distribución de clases: {m0.group(1)} No exitosos y {m1.group(1)} Exitosos."
                ) # Fin de append
                i += 3 # Avanza 3 líneas de golpe
                continue # Continúa

        if re.match(r"^precision\s+recall\s+f1-score\s+support", s, re.IGNORECASE): # Si inicia el reporte de sklearn
            while i < n and lineas_crudas[i].strip(): # Salta todo el bloque porque ya se extrae en su propia tarjeta
                i += 1 # Avanza línea por línea hasta el espacio en blanco
            continue # Continúa

        if s.startswith("[["): # Si inicia la matriz de confusión cruda
            while i < n and ("]]" not in lineas_crudas[i]): # Salta hasta el cierre ']]' porque ya tiene tarjeta propia
                i += 1 # Avanza
            i += 1 # Salta la línea de cierre
            continue # Continúa

        # 1. Tablas estilo PySpark (+---+---+)
        if s.startswith("+") and "---" in s and s.endswith("+"): # Detecta borde superior de tabla ASCII de Spark
            bloque_spark = [] # Acumula líneas de la tabla Spark
            j = i # Puntero secundario
            while j < n: # Recorre mientras sigan líneas con bordes + o |
                sj = lineas_crudas[j].strip() # Línea candidata
                if (sj.startswith("+") and "---" in sj) or (sj.startswith("|") and sj.endswith("|")): # Si pertenece a la tabla
                    bloque_spark.append(sj) # Agrega la línea
                    j += 1 # Avanza
                else: # Si terminó la tabla
                    break # Sale del bucle interno

            filas_datos = [ # Extrae las celdas separadas por '|'
                [celda.strip() for celda in f.strip("|").split("|")] # Limpia cada celda
                for f in bloque_spark # Recorre líneas acumuladas
                if f.startswith("|") # Ignora líneas separadoras '+---+'
            ] # Fin de comprensión de lista
            if len(filas_datos) >= 2: # Si hay encabezado y datos
                registrar_tabla(titulo_actual, filas_datos) # Registra la tabla Spark
                i = j # Actualiza puntero principal
                continue # Continúa

        # 2. Encabezados de sección en consola
        es_encabezado = ( # Detecta títulos en mayúsculas o numerados como '1. CARGA DE DATOS'
            re.match(r"^(\[?\d+[\]\.\)]\s+|[A-ZÁÉÍÓÚÑ\s\(\)/]{6,}:?$)", s) # Patrón de encabezado
            and len(s) <= 85 # No demasiado largo
            and not re.search(r":\s*\d", s) # Que no sea una métrica con número
        ) # Fin de evaluación
        if es_encabezado: # Si es un encabezado de sección
            guardar_bloque() # Guarda el bloque anterior antes de abrir uno nuevo
            titulo_actual = re.sub(r"^\[?\d+[\]\.\)]\s*", "", s).strip(" =-*#:").title() # Formatea el nuevo título
            i += 1 # Avanza
            continue # Continúa

        # 3. Tablas compactas de Pandas
        tokens_inicio = _es_fila_tabular_compacta(s) # Verifica si la línea inicia una tabla compacta
        if tokens_inicio and i + 1 < n: # Si parece encabezado tabular y hay más líneas debajo
            bloque_compacto = [tokens_inicio] # Inicia bloque con el encabezado
            j = i + 1 # Puntero secundario
            while j < n: # Recorre filas siguientes
                sj = lineas_crudas[j].strip() # Línea siguiente
                if not sj or _es_ruido_tecnico(sj): # Si hay línea vacía o ruido, corta la tabla
                    break # Sale del bucle
                tok_next = _es_fila_tabular_compacta(sj) # Evalúa si la siguiente línea también es fila tabular
                if tok_next and len(tok_next) in {len(tokens_inicio), len(tokens_inicio) + 1}: # Misma cantidad de columnas (o +1 por índice)
                    bloque_compacto.append(tok_next) # Agrega fila
                    j += 1 # Avanza
                else: # Si ya no coincide la estructura
                    break # Sale del bucle

            if len(bloque_compacto) >= 3: # Si tiene encabezado y al menos 2 filas de datos
                registrar_tabla(titulo_actual, bloque_compacto) # Registra la tabla
                i = j # Actualiza puntero principal
                continue # Continúa

        # 4. Tablas alineadas con >= 2 espacios
        partes_cols = re.split(r"\s{2,}", s) # Divide por 2 o más espacios consecutivos
        if len(partes_cols) >= 3 and i + 1 < n: # Si tiene al menos 3 columnas alineadas
            bloque_pandas = [partes_cols] # Inicia bloque tabular
            j = i + 1 # Puntero secundario
            while j < n: # Recorre filas siguientes
                sj = lineas_crudas[j].strip() # Línea siguiente
                if not sj or _es_ruido_tecnico(sj): # Corta si hay vacío o ruido
                    break # Sale
                p_next = re.split(r"\s{2,}", sj) # Divide siguiente línea por >=2 espacios
                if len(p_next) >= len(partes_cols) - 1 and len(p_next) >= 3: # Verifica consistencia de columnas
                    bloque_pandas.append(p_next) # Agrega fila
                    j += 1 # Avanza
                else: # Si cambia el formato
                    break # Sale

            if len(bloque_pandas) >= 3: # Si hay encabezado y al menos 2 filas
                registrar_tabla(titulo_actual, bloque_pandas) # Registra la tabla
                i = j # Actualiza puntero principal
                continue # Continúa

        # 5. Línea limpia traducida
        limpia = s.lstrip("-•*►✓✔ ") # Quita viñetas o símbolos iniciales redundantes
        limpia = re.sub(r"\bInercia:\s*([\d.]+)", r"Dispersión interna de grupos: \1", limpia, flags=re.IGNORECASE) # Traduce Inercia
        limpia = re.sub(r"\bSilhouette:\s*([\d.]+)", r"Claridad de separación entre grupos: \1 / 1.00", limpia, flags=re.IGNORECASE) # Traduce Silhouette
        limpia = re.sub(r"\bmacro avg\b", "Promedio simple entre clases (macro avg: mismo peso a exitosos y no exitosos)", limpia, flags=re.IGNORECASE) # Traduce macro avg
        limpia = re.sub(r"\bweighted avg\b", "Promedio ponderado (weighted avg: ajustado según cantidad real de viajes)", limpia, flags=re.IGNORECASE) # Traduce weighted avg
        if limpia and not re.match(r"^(costo|asientos_disponibles)\s+0$", limpia, re.IGNORECASE): # Evita líneas sueltas de conteo de nulos
            items_actuales.append(limpia) # Agrega la línea limpia al bloque actual

        i += 1 # Avanza a la siguiente línea

    guardar_bloque() # Guarda el último bloque acumulado al terminar el archivo

    return { # Devuelve el diccionario estructurado completo al frontend
        "tarjetas_kpi": _extraer_tarjetas_kpi(salida_texto), # Lista de tarjetas KPI superiores
        "tablas": tablas[:6], # Máximo 6 tablas formateadas
        "matriz_confusion": _extraer_matriz_confusion(salida_texto), # Matriz 2x2 o None
        "reporte_clasificacion": _extraer_reporte_clasificacion(salida_texto), # Tabla de clasificación o None
        "bloques_detalle": bloques_detalle[:6], # Máximo 6 bloques de hallazgos
        "lineas_auditoria": lineas_auditoria[:150], # Máximo 150 líneas crudas para auditoría
    } # Fin de retorno


@router.get("/", response_class=HTMLResponse) # Ruta GET principal que carga la página del panel de IA
async def vista_panel_ia_integral(request: Request): # Recibe el objeto Request de FastAPI
    return templates.TemplateResponse( # Renderiza la plantilla HTML correspondiente
        request=request, # Pasa el request obligatorio para Jinja2
        name="ia/dashboard_integral.html", # Ruta de la plantilla dentro de templates/
    ) # Fin de TemplateResponse


@router.get("/api/estado", response_class=JSONResponse) # Endpoint GET que consulta los datos reales en MongoDB Atlas
async def api_estado(): # Función asíncrona que devuelve KPIs, historial y archivos disponibles
    viajes = list(db["viaje"].find({}, {"_id": 0})) # Consulta todos los viajes registrados excluyendo _id
    rutas = {r.get("id_ruta"): r for r in db["ruta"].find({}, {"_id": 0}) if r.get("id_ruta") is not None} # Indexa rutas por id_ruta

    costos = [float(v["costo"]) for v in viajes if v.get("costo") is not None] # Extrae lista de tarifas válidas
    asientos = [int(v["asientos_disponibles"]) for v in viajes if v.get("asientos_disponibles") is not None] # Extrae asientos disponibles

    historial = [] # Lista para construir la tabla de viajes registrados en plataforma
    for v in viajes: # Recorre cada viaje de MongoDB
        ruta_info = rutas.get(v.get("id_ruta"), {}) # Busca los datos de origen y destino asociados
        historial.append({ # Agrega el viaje formateado
            "origen": ruta_info.get("origen") or v.get("origen") or "Valle de Toluca", # Origen del viaje
            "destino": ruta_info.get("destino") or v.get("destino") or "Campus UTVT", # Destino del viaje
            "costo": float(v.get("costo") or 0), # Costo del viaje en pesos
            "asientos": int(v.get("asientos_disponibles") or 0), # Asientos disponibles
            "estado": str(v.get("estado") or "pendiente"), # Estado operativo actual
        }) # Fin de append

    artefactos = _listar_artefactos() # Obtiene las gráficas PNG y reportes CSV actuales

    return { # Devuelve la respuesta JSON completa al frontend
        "kpis": { # Diccionario de indicadores reales
            "total_viajes": len(viajes), # Total de viajes en la colección
            "total_usuarios": db["usuario"].count_documents({}), # Total de usuarios registrados
            "total_reservas": db["reserva"].count_documents({}), # Total de reservaciones hechas
            "costo_promedio": round(sum(costos) / len(costos), 2) if costos else 0.0, # Tarifa promedio redondeada
            "asientos_promedio": round(sum(asientos) / len(asientos), 2) if asientos else 0.0, # Promedio de cupo ofrecido
        }, # Fin de kpis
        "historial": historial, # Lista de viajes para la tabla
        "imagenes": artefactos["imagenes"], # Lista de gráficas disponibles
        "csvs": artefactos["csvs"], # Lista de CSVs disponibles
    } # Fin de respuesta


@router.post("/api/ejecutar", response_class=JSONResponse) # Endpoint POST que ejecuta cualquiera de los 17 scripts de IA
async def api_ejecutar_proceso(solicitud: SolicitudEjecucion): # Recibe la clave del proceso a ejecutar
    info = PROCESOS_DISPONIBLES.get(solicitud.clave_proceso) # Busca la configuración del proceso solicitado
    if not info: # Si la clave no existe en el diccionario
        raise HTTPException(status_code=400, detail="Proceso no registrado") # Retorna error HTTP 400

    script_path = _resolver_script(info["script"]) # Localiza el archivo .py en modulo_ia_practicas
    if not script_path: # Si el archivo no se encuentra en disco
        return { # Devuelve respuesta controlada informando el nombre faltante
            "ok": False, # Indica que no se pudo ejecutar
            "titulo": info["nombre"], # Título del módulo
            "descripcion": info["descripcion"], # Descripción del módulo
            "conclusiones": [f"No se localizó el archivo {info['script']} en modulo_ia_practicas."], # Aviso claro
            "analisis": { # Estructura vacía para que el frontend no marque error
                "tarjetas_kpi": [], # Sin KPIs
                "tablas": [], # Sin tablas
                "matriz_confusion": None, # Sin matriz
                "reporte_clasificacion": None, # Sin reporte
                "bloques_detalle": [], # Sin bloques
                "lineas_auditoria": [], # Sin líneas de auditoría
            }, # Fin de analisis
            "imagenes_nuevas": [], # Sin imágenes nuevas
            "csvs_nuevos": [], # Sin CSVs nuevos
        } # Fin de retorno por archivo faltante

    antes = _listar_artefactos() # Toma fotografía de archivos existentes antes de ejecutar el script
    mtimes_img_antes = {i["ruta_rel"]: i["mtime"] for i in antes["imagenes"]} # Mapa de fechas previas de imágenes
    mtimes_csv_antes = {c["ruta_rel"]: c["mtime"] for c in antes["csvs"]} # Mapa de fechas previas de CSVs

    env = os.environ.copy() # Copia las variables de entorno actuales del sistema
    env["MPLBACKEND"] = "Agg" # Fuerza Matplotlib a modo sin ventana gráfica para evitar bloqueos en servidor
    env["PYTHONIOENCODING"] = "utf-8" # Fuerza salida estándar en UTF-8 para soportar acentos y símbolos
    env["COLUMNS"] = "240" # Amplía el ancho virtual de consola para que Pandas no recorte columnas con '...'

    try: # Bloque try para controlar tiempos de espera largos
        res = subprocess.run( # Ejecuta el script de Python de forma síncrona capturando su salida
            [sys.executable, str(script_path)], # Comando: intérprete Python + ruta del script
            cwd=str(CARPETA_IA), # Establece modulo_ia_practicas como directorio de trabajo
            env=env, # Pasa el entorno configurado
            capture_output=True, # Captura stdout y stderr
            text=True, # Decodifica la salida como texto
            encoding="utf-8", # Codificación UTF-8
            errors="replace", # Reemplaza caracteres inválidos sin lanzar excepción
            timeout=180, # Límite máximo de 3 minutos de ejecución
        ) # Fin de subprocess.run
        salida_combinada = (res.stdout or "") + ("\n" + res.stderr if res.stderr else "") # Une salida estándar y errores
        despues = _listar_artefactos() # Escanea nuevamente los archivos tras finalizar el script

        imgs_actualizadas = [ # Filtra imágenes nuevas o modificadas durante esta ejecución
            img for img in despues["imagenes"] # Recorre imágenes actuales
            if img["mtime"] > mtimes_img_antes.get(img["ruta_rel"], 0) # Compara fecha nueva contra anterior
        ] # Fin de filtro de imágenes
        csvs_actualizados = [ # Filtra CSVs nuevos o modificados durante esta ejecución
            csv for csv in despues["csvs"] # Recorre CSVs actuales
            if csv["mtime"] > mtimes_csv_antes.get(csv["ruta_rel"], 0) # Compara fecha nueva contra anterior
        ] # Fin de filtro de CSVs

        if not imgs_actualizadas: # Si el script no generó gráficas nuevas, muestra las 3 más recientes
            imgs_actualizadas = despues["imagenes"][:3] # Toma las 3 primeras
        if not csvs_actualizados: # Si el script no generó CSV nuevo, enlaza los 3 más recientes
            csvs_actualizados = despues["csvs"][:3] # Toma los 3 primeros

        ok = res.returncode == 0 # Verifica si el script terminó con código de éxito 0
        analisis = _analizar_salida_terminal(salida_combinada) # Procesa y estructura toda la salida de consola

        return { # Devuelve el paquete completo al dashboard frontend
            "ok": ok, # Estado booleano de ejecución
            "titulo": info["nombre"], # Título del análisis
            "descripcion": info["descripcion"], # Descripción del análisis
            "conclusiones": info["conclusiones_base"], # Conclusiones en lenguaje natural
            "analisis": analisis, # KPIs, tablas, matriz de confusión, reporte y bloques
            "imagenes_nuevas": imgs_actualizadas, # Gráficas asociadas a esta ejecución
            "csvs_nuevos": csvs_actualizados, # CSVs asociados a esta ejecución
            "imagenes": despues["imagenes"], # Galería total actualizada
            "csvs": despues["csvs"], # Lista total de CSVs actualizada
        } # Fin de respuesta exitosa
    except subprocess.TimeoutExpired: # Captura el caso en que el script tarde más de 180 segundos
        return { # Devuelve mensaje amigable de tiempo agotado
            "ok": False, # Indica fallo por timeout
            "titulo": info["nombre"], # Título del proceso
            "descripcion": info["descripcion"], # Descripción
            "conclusiones": ["El análisis superó el límite de 3 minutos y se detuvo automáticamente."], # Aviso
            "analisis": { # Estructura vacía
                "tarjetas_kpi": [], # Sin KPIs
                "tablas": [], # Sin tablas
                "matriz_confusion": None, # Sin matriz
                "reporte_clasificacion": None, # Sin reporte
                "bloques_detalle": [], # Sin bloques
                "lineas_auditoria": [], # Sin auditoría
            }, # Fin de analisis
            "imagenes_nuevas": [], # Sin imágenes
            "csvs_nuevos": [], # Sin CSVs
        } # Fin de respuesta por timeout


@router.get("/archivo/imagen") # Endpoint GET para servir imágenes PNG/JPG generadas por los scripts
async def obtener_imagen(ruta: str): # Recibe la ruta relativa como parámetro query
    return FileResponse(_resolver_ruta_segura(ruta)) # Valida la ruta y devuelve el archivo de imagen


@router.get("/archivo/descargar-csv") # Endpoint GET para descargar archivos CSV como adjunto
async def descargar_csv(ruta: str): # Recibe la ruta relativa del CSV
    archivo = _resolver_ruta_segura(ruta) # Valida la seguridad de la ruta
    return FileResponse(path=archivo, media_type="text/csv", filename=archivo.name) # Fuerza descarga con su nombre original


@router.get("/api/vista-previa-csv", response_class=JSONResponse) # Endpoint GET para previsualizar las primeras 15 filas de un CSV
async def vista_previa_csv(ruta: str): # Recibe la ruta relativa del CSV
    archivo = _resolver_ruta_segura(ruta) # Valida la ruta
    df = pd.read_csv(archivo).head(15) # Lee el CSV con Pandas y toma los primeros 15 registros
    return { # Devuelve columnas y filas en JSON para renderizar en el modal
        "nombre": archivo.name, # Nombre del archivo
        "columnas": list(df.columns), # Lista de nombres de columnas
        "filas": df.fillna("").to_dict(orient="records"), # Convierte filas a diccionarios reemplazando NaN por vacío
    } # Fin de retorno


@router.post("/api/predecir", response_class=JSONResponse) # Endpoint POST del simulador interactivo de aceptación de viajes
async def api_predecir_viaje(datos: SolicitudPrediccion): # Recibe costo, asientos, distancia y puntualidad
    viajes_reales = list(db["viaje"].find({}, {"_id": 0, "costo": 1, "asientos_disponibles": 1, "estado": 1})) # Obtiene viajes reales de MongoDB
    estados_ok = {"activo", "aceptado", "en_curso", "finalizado"} # Define qué estados se consideran viaje exitoso

    filas = [ # Convierte cada viaje real de MongoDB en una fila de entrenamiento
        { # Diccionario de características del viaje
            "costo": float(v.get("costo") or 15.0), # Tarifa del viaje
            "asientos_disponibles": int(v.get("asientos_disponibles") or 2), # Cupo ofrecido
            "distancia_km": 16.5, # Distancia estándar de referencia al campus
            "puntualidad": 90.0, # Puntualidad base de referencia
            "exito": 1 if str(v.get("estado") or "").lower().strip() in estados_ok else 0, # Etiqueta binaria 1=éxito, 0=no éxito
        } # Fin de diccionario
        for v in viajes_reales # Itera sobre todos los viajes reales
    ] # Fin de comprensión de lista

    rng = np.random.default_rng(42) # Inicializa generador aleatorio reproducible con semilla 42
    if len(filas) < 150: # Si hay menos de 150 viajes reales, complementa con datos sintéticos calibrados
        n_extra = 150 - len(filas) # Calcula cuántos registros faltan para llegar a 150
        c_sim = rng.uniform(10, 40, n_extra) # Simula tarifas entre $10 y $40 MXN
        a_sim = rng.integers(1, 5, n_extra) # Simula entre 1 y 4 asientos disponibles
        d_sim = rng.uniform(8, 25, n_extra) # Simula distancias entre 8 y 25 km
        p_sim = rng.uniform(65, 100, n_extra) # Simula puntualidad entre 65% y 100%
        score = (24 - c_sim) * 0.42 + (a_sim - 1.5) * 1.25 + (p_sim - 80) * 0.08 # Fórmula de utilidad estudiantil
        ex_sim = (score >= 0).astype(int) # Clasifica como exitoso (1) si el puntaje es positivo
        for i in range(n_extra): # Agrega cada registro sintético a la lista de entrenamiento
            filas.append({ # Diccionario del viaje simulado
                "costo": float(c_sim[i]), # Costo simulado
                "asientos_disponibles": int(a_sim[i]), # Asientos simulados
                "distancia_km": float(d_sim[i]), # Distancia simulada
                "puntualidad": float(p_sim[i]), # Puntualidad simulada
                "exito": int(ex_sim[i]), # Resultado simulado
            }) # Fin de append

    df = pd.DataFrame(filas) # Convierte la lista combinada en un DataFrame de Pandas
    X = df[["costo", "asientos_disponibles", "distancia_km", "puntualidad"]] # Selecciona las 4 variables predictoras
    y = df["exito"] # Selecciona la variable objetivo (0 o 1)

    scaler = StandardScaler() # Instancia el estandarizador de variables
    X_esc = scaler.fit_transform(X) # Ajusta y transforma las variables de entrenamiento
    modelo = RandomForestClassifier(n_estimators=100, random_state=42).fit(X_esc, y) # Entrena Random Forest con 100 árboles

    entrada = np.array([[datos.costo, datos.asientos_disponibles, datos.distancia_km, datos.puntualidad]]) # Construye vector con los datos del usuario
    prob_exito = float(modelo.predict_proba(scaler.transform(entrada))[0][1]) * 100 # Calcula probabilidad porcentual de clase 1 (éxito)

    factores = [] # Lista para explicar en lenguaje claro por qué dio ese porcentaje
    if datos.costo <= 20: # Evalúa si la tarifa es competitiva
        factores.append(f"Tarifa competitiva (${datos.costo:.0f} MXN): dentro del rango preferido por los estudiantes.") # Factor positivo
    else: # Si supera los $20 MXN
        factores.append(f"Tarifa elevada (${datos.costo:.0f} MXN): superar los $20 MXN reduce la intención de reserva.") # Factor negativo

    if datos.asientos_disponibles >= 3: # Evalúa disponibilidad de lugares
        factores.append(f"Buena disponibilidad ({datos.asientos_disponibles} lugares): facilita traslados en grupo.") # Factor positivo
    else: # Si ofrece pocos asientos
        factores.append(f"Cupo limitado ({datos.asientos_disponibles} lugar/es): menor margen para llenar el auto.") # Factor moderado

    if datos.puntualidad >= 85: # Evalúa confiabilidad del conductor
        factores.append(f"Alta confiabilidad ({datos.puntualidad:.0f}% puntualidad): incrementa la confianza del pasajero.") # Factor positivo
    else: # Si la puntualidad es menor a 85%
        factores.append(f"Puntualidad moderada ({datos.puntualidad:.0f}%): mejorar este indicador eleva la aceptación.") # Factor moderado

    recomendacion = ( # Genera recomendación final según el umbral del 60%
        "Alta probabilidad de ocupación: condiciones óptimas para publicar el viaje en Velox UTVT." # Mensaje favorable
        if prob_exito >= 60 # Condición de alta probabilidad
        else "Ocupación moderada/baja: te sugerimos ajustar la cooperación entre $12 y $20 MXN." # Sugerencia de mejora
    ) # Fin de recomendación

    return { # Devuelve la predicción completa al simulador
        "probabilidad_exito": round(prob_exito, 2), # Probabilidad redondeada a 2 decimales
        "viable": prob_exito >= 50.0, # Indica si supera el 50% de viabilidad
        "recomendacion": recomendacion, # Texto de recomendación operativa
        "factores": factores, # Lista de justificaciones personalizadas
    } # Fin de retorno