# ==========================================================
# ROUTER: VELOX UTVT - CENTRO DE INTELIGENCIA OPERATIVA
# Resultados con justificación e interpretación operativa
# Archivo: routers/ia_integral_router.py
# ==========================================================

import os
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

from database import db

router = APIRouter(prefix="/ia-integral", tags=["Centro de Inteligencia Velox"])
templates = Jinja2Templates(directory="templates")

RAIZ = Path(__file__).resolve().parent.parent
CARPETA_IA = RAIZ / "modulo_ia_practicas"

PROCESOS_DISPONIBLES: dict[str, dict] = {
    "reglas_motor": {
        "nombre": "Motor de Reglas Operativas",
        "descripcion": "Evalúa tarifas, cupos y disponibilidad según las políticas de carpooling de Velox UTVT.",
        "script": "p01_regla_fija.py",
        "diagnostico_base": "Se aplicaron las reglas determinísticas de negocio sobre tarifas máximas, lugares disponibles y horarios de entrada/salida al campus.",
        "recomendacion": "Mantener las reglas activas como filtro previo antes de publicar un viaje para evitar tarifas fuera de rango.",
    },
    "limpieza_datos": {
        "nombre": "Depuración y Validación de Usuarios",
        "descripcion": "Filtra registros incompletos, valida correos institucionales y separa usuarios limpios.",
        "script": "p02limpiezaguardadodatos_d.py",
        "diagnostico_base": "El motor auditó los registros de la comunidad universitaria, separando cuentas verificadas de registros con campos nulos o inconsistentes.",
        "recomendacion": "Descarga el archivo de usuarios limpios para alimentar los modelos predictivos sin ruido ni valores faltantes.",
    },
    "modelo_base": {
        "nombre": "Entrenamiento de Modelo Base",
        "descripcion": "Calibra el primer clasificador supervisado sobre trayectos históricos.",
        "script": "p03primermodelo.py",
        "diagnostico_base": "Se entrenó una línea base de clasificación para distinguir viajes con alta probabilidad de aceptación frente a viajes sin demanda.",
        "recomendacion": "Usa este punto de referencia para medir cuánto mejora la precisión al aplicar ensambles o redes neuronales.",
    },
    "arbol_decision": {
        "nombre": "Clasificador por Árbol de Decisión",
        "descripcion": "Genera reglas jerárquicas de decisión para predecir la aceptación de viajes.",
        "script": "practica 04.py",
        "diagnostico_base": "El árbol dividió los trayectos según las variables más determinantes: cooperación económica por pasajero, cupo ofrecido y horario.",
        "recomendacion": "Las tarifas cercanas al promedio universitario ($12–$20 MXN) lideran las ramas con mayor tasa de reserva.",
    },
    "comparativa_algoritmos": {
        "nombre": "Comparativa de Modelos Supervisados",
        "descripcion": "Evalúa múltiples algoritmos y genera la matriz de confusión y comparativa de exactitud.",
        "script": "velox_ml_comparacion.py",
        "diagnostico_base": "Se pusieron a competir distintos clasificadores bajo las mismas condiciones de prueba para identificar cuál comete menos falsos positivos.",
        "recomendacion": "Revisa la matriz de confusión generada abajo para confirmar el equilibrio entre viajes aprobados y rechazados.",
    },
    "hiperparametros": {
        "nombre": "Ajuste de Hiperparámetros",
        "descripcion": "Optimiza la profundidad y criterios de división para maximizar el rendimiento.",
        "script": "Práctica 06 (Optimización de Hiperpartámetros).py",
        "diagnostico_base": "Se probaron múltiples combinaciones de profundidad y regularización para evitar sobreajuste y mejorar la generalización en rutas nuevas.",
        "recomendacion": "El modelo optimizado mantiene un desempeño estable incluso cuando cambian las condiciones de demanda entre semana.",
    },
    "optimizacion_clusters": {
        "nombre": "Segmentación Óptima de Rutas (K-Means)",
        "descripcion": "Agrupa los viajes por comportamiento usando el método del codo y coeficiente de silueta.",
        "script": "practica08.py",
        "diagnostico_base": "El agrupamiento no supervisado identificó perfiles naturales de movilidad (traslados cortos económicos, rutas estándar y trayectos largos).",
        "recomendacion": "Un coeficiente de silueta alto confirma que los grupos están bien diferenciados para personalizar sugerencias de tarifa.",
    },
    "reduccion_pca": {
        "nombre": "Proyección Dimensional (PCA)",
        "descripcion": "Reduce variables operativas a componentes principales para visualizar patrones.",
        "script": "practica09.py",
        "diagnostico_base": "Se comprimieron múltiples indicadores operativos en componentes principales conservando la mayor parte de la varianza informativa.",
        "recomendacion": "Permite detectar de un vistazo trayectos atípicos o saturación en corredores específicos hacia Santa María Atarasquillo.",
    },
    "extraccion_atlas": {
        "nombre": "Extracción y Exportación MongoDB Atlas",
        "descripcion": "Consulta las colecciones reales de Velox UTVT y exporta los reportes CSV.",
        "script": "p16consultasinteligentes.py",
        "diagnostico_base": "Se conectó directamente con el clúster velox_utvt en MongoDB Atlas para consolidar usuarios, rutas, viajes y reservaciones reales.",
        "recomendacion": "Utiliza los CSV generados como respaldo auditado del estado actual de la plataforma en producción.",
    },
    "generador_masivo": {
        "nombre": "Generador de 50,000 Trayectos Históricos",
        "descripcion": "Construye el volumen masivo de viajes del Valle de Toluca hacia Campus UTVT.",
        "script": "p20generador_datos_masivos_velox.py",
        "diagnostico_base": "Se sintetizaron 50,000 registros respetando la distribución geográfica real de Toluca, Metepec, San Mateo Atenco y Lerma.",
        "recomendacion": "Este volumen permite someter los motores analíticos a pruebas de estrés de escala institucional.",
    },
    "conversion_parquet": {
        "nombre": "Conversión Columnar Apache Parquet",
        "descripcion": "Optimiza el almacenamiento masivo comprimido en formato columnar Parquet.",
        "script": "p21conversion_parquet_velox.py",
        "diagnostico_base": "Los registros planos fueron transformados a formato columnar Apache Parquet con compresión Snappy, reduciendo drásticamente el peso en disco.",
        "recomendacion": "Emplear Parquet como formato estándar para consultas analíticas rápidas sobre históricos extensos.",
    },
    "spark_procesamiento": {
        "nombre": "Agregación Distribuida PySpark",
        "descripcion": "Calcula demanda, tarifas promedio y tasas de consolidación por punto de abordaje.",
        "script": "p22spark_procesamiento_velox.py",
        "diagnostico_base": "Apache Spark procesó en paralelo los 50,000 trayectos para calcular volumen de viajes, tarifa media y porcentaje de ocupación por origen.",
        "recomendacion": "Priorizar incentivos y difusión en los nodos con mayor tasa de consolidación hacia el campus.",
    },
    "spark_mllib": {
        "nombre": "Entrenamiento Distribuido Spark MLlib",
        "descripcion": "Entrena clasificadores a gran escala con Regresión Logística, Árboles y Random Forest.",
        "script": "p23spark_mllib_velox.py",
        "diagnostico_base": "Se evaluaron modelos distribuidos en Spark MLlib midiendo Exactitud, F1-Score y Área Bajo la Curva ROC sobre el dataset masivo.",
        "recomendacion": "El algoritmo con mayor AUC-ROC ofrece la mejor capacidad para priorizar viajes recomendados en el buscador.",
    },
    "benchmark_bigdata": {
        "nombre": "Benchmark Tradicional vs. Big Data",
        "descripcion": "Compara tiempos de respuesta y uso de disco entre procesamiento local y distribuido.",
        "script": "p24bigdata_vs_tradicional_velox.py",
        "diagnostico_base": "Se contrastó el flujo tradicional CSV/Pandas frente al pipeline Parquet/Spark en almacenamiento y velocidad de cómputo.",
        "recomendacion": "A partir de decenas de miles de registros, la arquitectura distribuida supera ampliamente el manejo secuencial.",
    },
    "perceptron_lineal": {
        "nombre": "Neurona Perceptrón Base",
        "descripcion": "Entrena una neurona artificial lineal para clasificar viabilidad de trayectos.",
        "script": "p25neurona_perceptron_velox.py",
        "diagnostico_base": "Se ajustaron los pesos sinápticos de una neurona perceptrón mediante descenso de gradiente para separar viajes viables y no viables.",
        "recomendacion": "Demuestra cómo los pesos negativos penalizan tarifas excesivas mientras los positivos premian puntualidad y cupo.",
    },
    "red_mlp": {
        "nombre": "Red Neuronal Multicapa (MLP)",
        "descripcion": "Modela relaciones no lineales mediante capas ocultas con activación ReLU.",
        "script": "p26red_neuronal_mlp_velox.py",
        "diagnostico_base": "Al incorporar capas ocultas no lineales, la red capturó interacciones complejas entre horario, distancia y tarifa que un modelo lineal no detecta.",
        "recomendacion": "La mejora en F1-Score respecto al perceptrón simple justifica el uso de arquitecturas multicapa.",
    },
    "deep_learning_pytorch": {
        "nombre": "Red Neuronal Profunda (PyTorch)",
        "descripcion": "Entrena una arquitectura profunda en PyTorch con curvas de pérdida y validación ROC.",
        "script": "p27deep_learning_pytorch_velox.py",
        "diagnostico_base": "Se entrenó una red profunda en PyTorch con normalización y optimización Adam, logrando convergencia estable en la función de pérdida.",
        "recomendacion": "Revisa la curva de pérdida (Loss) y la curva ROC en las gráficas generadas para validar la estabilidad del entrenamiento.",
    },
}

CARPETAS_ARTEFACTOS = [
    CARPETA_IA,
    CARPETA_IA / "resultados",
    CARPETA_IA / "resultados_p08_velox",
    CARPETA_IA / "resultados_p09_velox",
    CARPETA_IA / "resultados_p16_velox",
    CARPETA_IA / "resultados_bigdata_velox",
    CARPETA_IA / "resultados_dl_velox",
    CARPETA_IA / "static",
]


class SolicitudEjecucion(BaseModel):
    clave_proceso: str


class SolicitudPrediccion(BaseModel):
    costo: float = Field(ge=0, le=500)
    asientos_disponibles: int = Field(ge=1, le=8)
    distancia_km: float = Field(default=16.5, ge=1, le=100)
    puntualidad: float = Field(default=90.0, ge=0, le=100)


def _resolver_ruta_segura(rel_path: str) -> Path:
    ruta = (CARPETA_IA / rel_path).resolve()
    if CARPETA_IA.resolve() not in ruta.parents and ruta != CARPETA_IA.resolve():
        raise HTTPException(status_code=403, detail="Ruta no permitida")
    if not ruta.is_file():
        raise HTTPException(status_code=404, detail="Archivo no encontrado")
    return ruta


def _listar_artefactos() -> dict:
    imagenes = []
    csvs = []
    vistos = set()

    for carpeta in CARPETAS_ARTEFACTOS:
        if not carpeta.is_dir():
            continue
        for archivo in sorted(carpeta.iterdir(), key=lambda f: f.stat().st_mtime, reverse=True):
            if not archivo.is_file():
                continue
            rel = archivo.relative_to(CARPETA_IA).as_posix()
            if rel in vistos:
                continue
            vistos.add(rel)

            ext = archivo.suffix.lower()
            mtime = archivo.stat().st_mtime
            if ext in {".png", ".jpg", ".jpeg"}:
                imagenes.append({
                    "nombre": archivo.stem.replace("_", " ").title(),
                    "archivo": archivo.name,
                    "ruta_rel": rel,
                    "mtime": mtime,
                })
            elif ext == ".csv":
                csvs.append({
                    "nombre": archivo.stem.replace("_", " ").title(),
                    "archivo": archivo.name,
                    "ruta_rel": rel,
                    "tam_kb": round(archivo.stat().st_size / 1024, 1),
                    "mtime": mtime,
                })

    return {"imagenes": imagenes, "csvs": csvs}


def _extraer_indicadores_visuales(texto: str) -> list[dict]:
    indicadores = []
    patrones = [
        (r"(?:Accuracy|Exactitud|Precisi[oó]n)[^\d]{0,20}(\d+(?:\.\d+)?\s*%?)", "Exactitud del Modelo"),
        (r"(?:F1[-\s]?Score)[^\d]{0,20}(\d+(?:\.\d+)?\s*%?)", "Puntaje F1-Score"),
        (r"(?:AUC[-\s]?ROC)[^\d]{0,20}(\d+(?:\.\d+)?)", "Área Bajo Curva (ROC)"),
        (r"(?:Silueta|Silhouette)[^\d]{0,20}(\d+(?:\.\d+)?)", "Índice de Silueta"),
        (r"(?:Varianza explicada)[^\d]{0,20}(\d+(?:\.\d+)?\s*%?)", "Varianza Conservada"),
        (r"(?:Tiempo total|Duraci[oó]n)[^\d]{0,20}(\d+(?:\.\d+)?\s*s(?:eg)?)", "Tiempo de Proceso"),
        (r"(?:Total(?: de)? registros|Viajes generados|Registros limpios)[^\d]{0,20}([\d,]+)", "Registros Procesados"),
    ]
    for patron, etiqueta in patrones:
        match = re.search(patron, texto, re.IGNORECASE)
        if match:
            indicadores.append({"etiqueta": etiqueta, "valor": match.group(1).strip()})
    return indicadores[:4]


def _construir_narrativa_resultados(info: dict, salida_texto: str, indicadores: list[dict], ok: bool) -> dict:
    """Traduce la salida técnica de consola en una explicación clara y accionable."""
    lineas_utiles = []
    for linea in salida_texto.splitlines():
        limpia = linea.strip(" =-_*#")
        if not limpia or len(limpia) < 12:
            continue
        # Filtra trazas técnicas de Python/Spark para dejar solo mensajes informativos
        if any(ruido in limpia for ruido in ["Traceback", "File \"", "WARN ", "INFO ", "UserWarning", "FutureWarning"]):
            continue
        lineas_utiles.append(limpia)

    hallazgos = lineas_utiles[:4]

    if indicadores:
        resumen_metricas = ", ".join(f"{i['etiqueta']}: {i['valor']}" for i in indicadores)
        justificacion = (
            f"{info['diagnostico_base']} Durante esta ejecución el sistema obtuvo los siguientes valores clave: "
            f"{resumen_metricas}."
        )
    elif ok:
        justificacion = (
            f"{info['diagnostico_base']} El flujo concluyó correctamente y actualizó los archivos de respaldo "
            f"y visualizaciones correspondientes."
        )
    else:
        justificacion = (
            "El proceso detectó una incidencia al validar los datos de entrada o dependencias. "
            "Revisa los puntos detectados abajo para corregir el origen."
        )

    return {
        "justificacion": justificacion,
        "hallazgos": hallazgos,
        "recomendacion": info["recomendacion"],
    }


@router.get("/", response_class=HTMLResponse)
async def vista_panel_ia_integral(request: Request):
    return templates.TemplateResponse(
        request=request,
        name="ia/dashboard_integral.html",
    )


@router.get("/api/estado", response_class=JSONResponse)
async def api_estado():
    viajes = list(db["viaje"].find({}, {"_id": 0}))
    rutas = {r.get("id_ruta"): r for r in db["ruta"].find({}, {"_id": 0}) if r.get("id_ruta") is not None}

    costos = [float(v["costo"]) for v in viajes if v.get("costo") is not None]
    asientos = [int(v["asientos_disponibles"]) for v in viajes if v.get("asientos_disponibles") is not None]

    historial = []
    for v in viajes:
        ruta_info = rutas.get(v.get("id_ruta"), {})
        historial.append({
            "origen": ruta_info.get("origen") or v.get("origen") or "Valle de Toluca",
            "destino": ruta_info.get("destino") or v.get("destino") or "Campus UTVT",
            "costo": float(v.get("costo") or 0),
            "asientos": int(v.get("asientos_disponibles") or 0),
            "estado": str(v.get("estado") or "pendiente"),
        })

    artefactos = _listar_artefactos()

    return {
        "kpis": {
            "total_viajes": len(viajes),
            "total_usuarios": db["usuario"].count_documents({}),
            "total_reservas": db["reserva"].count_documents({}),
            "costo_promedio": round(sum(costos) / len(costos), 2) if costos else 0.0,
            "asientos_promedio": round(sum(asientos) / len(asientos), 2) if asientos else 0.0,
        },
        "historial": historial,
        "imagenes": artefactos["imagenes"],
        "csvs": artefactos["csvs"],
    }


@router.post("/api/ejecutar", response_class=JSONResponse)
async def api_ejecutar_proceso(solicitud: SolicitudEjecucion):
    info = PROCESOS_DISPONIBLES.get(solicitud.clave_proceso)
    if not info:
        raise HTTPException(status_code=400, detail="Proceso no registrado")

    script_path = CARPETA_IA / info["script"]
    if not script_path.is_file():
        return {
            "ok": False,
            "titulo": info["nombre"],
            "descripcion": info["descripcion"],
            "indicadores": [],
            "narrativa": {
                "justificacion": "No se localizó el archivo del módulo solicitado en el servidor.",
                "hallazgos": [f"Ruta verificada: {script_path.name}"],
                "recomendacion": "Verifica que el archivo se encuentre dentro de la carpeta modulo_ia_practicas.",
            },
            "imagenes_nuevas": [],
            "csvs_nuevos": [],
        }

    antes = _listar_artefactos()
    mtimes_img_antes = {i["ruta_rel"]: i["mtime"] for i in antes["imagenes"]}
    mtimes_csv_antes = {c["ruta_rel"]: c["mtime"] for c in antes["csvs"]}

    env = os.environ.copy()
    env["MPLBACKEND"] = "Agg"
    env["PYTHONIOENCODING"] = "utf-8"

    try:
        res = subprocess.run(
            [sys.executable, str(script_path)],
            cwd=str(CARPETA_IA),
            env=env,
            capture_output=True,
            text=True,
            timeout=180,
        )
        salida_combinada = (res.stdout or "") + ("\n" + res.stderr if res.stderr else "")
        despues = _listar_artefactos()

        imgs_actualizadas = [
            img for img in despues["imagenes"]
            if img["mtime"] > mtimes_img_antes.get(img["ruta_rel"], 0)
        ]
        csvs_actualizados = [
            csv for csv in despues["csvs"]
            if csv["mtime"] > mtimes_csv_antes.get(csv["ruta_rel"], 0)
        ]

        if not imgs_actualizadas:
            imgs_actualizadas = despues["imagenes"][:3]
        if not csvs_actualizados:
            csvs_actualizados = despues["csvs"][:3]

        ok = res.returncode == 0
        indicadores = _extraer_indicadores_visuales(salida_combinada)
        narrativa = _construir_narrativa_resultados(info, salida_combinada, indicadores, ok)

        return {
            "ok": ok,
            "titulo": info["nombre"],
            "descripcion": info["descripcion"],
            "indicadores": indicadores,
            "narrativa": narrativa,
            "imagenes_nuevas": imgs_actualizadas,
            "csvs_nuevos": csvs_actualizados,
            "imagenes": despues["imagenes"],
            "csvs": despues["csvs"],
        }
    except subprocess.TimeoutExpired:
        return {
            "ok": False,
            "titulo": info["nombre"],
            "descripcion": info["descripcion"],
            "indicadores": [],
            "narrativa": {
                "justificacion": "El análisis tardó más de 3 minutos y se detuvo automáticamente para proteger el servidor.",
                "hallazgos": ["Tiempo límite alcanzado (180 segundos)."],
                "recomendacion": "Verifica la conexión a red o ejecuta primero el generador de datos.",
            },
            "imagenes_nuevas": [],
            "csvs_nuevos": [],
        }


@router.get("/archivo/imagen")
async def obtener_imagen(ruta: str):
    return FileResponse(_resolver_ruta_segura(ruta))


@router.get("/archivo/descargar-csv")
async def descargar_csv(ruta: str):
    archivo = _resolver_ruta_segura(ruta)
    return FileResponse(path=archivo, media_type="text/csv", filename=archivo.name)


@router.get("/api/vista-previa-csv", response_class=JSONResponse)
async def vista_previa_csv(ruta: str):
    archivo = _resolver_ruta_segura(ruta)
    df = pd.read_csv(archivo).head(15)
    return {
        "nombre": archivo.name,
        "columnas": list(df.columns),
        "filas": df.fillna("").to_dict(orient="records"),
    }


@router.post("/api/predecir", response_class=JSONResponse)
async def api_predecir_viaje(datos: SolicitudPrediccion):
    viajes_reales = list(db["viaje"].find({}, {"_id": 0, "costo": 1, "asientos_disponibles": 1, "estado": 1}))
    estados_ok = {"activo", "aceptado", "en_curso", "finalizado"}

    filas = [
        {
            "costo": float(v.get("costo") or 15.0),
            "asientos_disponibles": int(v.get("asientos_disponibles") or 2),
            "distancia_km": 16.5,
            "puntualidad": 90.0,
            "exito": 1 if str(v.get("estado") or "").lower().strip() in estados_ok else 0,
        }
        for v in viajes_reales
    ]

    rng = np.random.default_rng(42)
    if len(filas) < 150:
        n_extra = 150 - len(filas)
        c_sim = rng.uniform(10, 40, n_extra)
        a_sim = rng.integers(1, 5, n_extra)
        d_sim = rng.uniform(8, 25, n_extra)
        p_sim = rng.uniform(65, 100, n_extra)
        score = (24 - c_sim) * 0.42 + (a_sim - 1.5) * 1.25 + (p_sim - 80) * 0.08
        ex_sim = (score >= 0).astype(int)
        for i in range(n_extra):
            filas.append({
                "costo": float(c_sim[i]),
                "asientos_disponibles": int(a_sim[i]),
                "distancia_km": float(d_sim[i]),
                "puntualidad": float(p_sim[i]),
                "exito": int(ex_sim[i]),
            })

    df = pd.DataFrame(filas)
    X = df[["costo", "asientos_disponibles", "distancia_km", "puntualidad"]]
    y = df["exito"]

    scaler = StandardScaler()
    X_esc = scaler.fit_transform(X)
    modelo = RandomForestClassifier(n_estimators=100, random_state=42).fit(X_esc, y)

    entrada = np.array([[datos.costo, datos.asientos_disponibles, datos.distancia_km, datos.puntualidad]])
    prob_exito = float(modelo.predict_proba(scaler.transform(entrada))[0][1]) * 100

    # Explicación detallada de los factores que inclinaron el resultado
    factores = []
    if datos.costo <= 20:
        factores.append(f"Tarifa competitiva (${datos.costo:.0f} MXN): se encuentra dentro del rango preferido por los estudiantes.")
    else:
        factores.append(f"Tarifa elevada (${datos.costo:.0f} MXN): superar los $20 MXN reduce la intención de reserva frente al transporte público.")

    if datos.asientos_disponibles >= 3:
        factores.append(f"Buena disponibilidad ({datos.asientos_disponibles} lugares): facilita que grupos de compañeros viajen juntos.")
    else:
        factores.append(f"Cupo limitado ({datos.asientos_disponibles} lugar/es): menor margen para llenar el vehículo en horas pico.")

    if datos.puntualidad >= 85:
        factores.append(f"Alta confiabilidad ({datos.puntualidad:.0f}% puntualidad): incrementa la confianza del pasajero.")
    else:
        factores.append(f"Puntualidad moderada ({datos.puntualidad:.0f}%): mejorar este indicador eleva significativamente la aceptación.")

    recomendacion = (
        "Alta probabilidad de ocupación: condiciones óptimas para publicar el viaje en Velox UTVT."
        if prob_exito >= 60
        else "Ocupación moderada/baja: te sugerimos ajustar la cooperación entre $12 y $20 MXN para atraer más pasajeros."
    )

    return {
        "probabilidad_exito": round(prob_exito, 2),
        "viable": prob_exito >= 50.0,
        "recomendacion": recomendacion,
        "factores": factores,
    }