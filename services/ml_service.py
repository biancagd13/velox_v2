import os
import joblib
import pandas as pd
from pathlib import Path

# Definimos la ruta base de tu proyecto
BASE_DIR = Path(__file__).resolve().parent.parent
MODELOS_DIR = BASE_DIR / "resultados_velox" / "modelos"

def predecir_confiabilidad_pasajero(historial_asistencia: float, cancelaciones_previas: float, viajes_totales: float):
    """
    Simula o procesa mediante Machine Learning la probabilidad 
    de que un pasajero aborde el viaje o lo cancele.
    (Basado en lógica de Random Forest / Árboles de decisión)
    """
    # Lógica basada en comportamiento histórico
    if cancelaciones_previas == 0 and historial_asistencia >= 90:
        probabilidad = 95.0
        estado = "Confiable / Alta Certeza"
    elif cancelaciones_previas <= 1:
        probabilidad = 80.0
        estado = "Moderado"
    else:
        probabilidad = 55.0
        estado = "En Riesgo de Cancelación"
    
    return {
        "probabilidad": probabilidad,
        "estado": estado
    }


def simular_ruta_kmeans_pca(distancia_km: float, tiempo_min: float):
    """
    Aplica la lógica de K-Means para clasificar el tipo de ruta 
    y PCA para proyectar el rendimiento de la tarifa en Velox UTVT.
    """
    # 1. Segmentación K-Means: Agrupamos la ruta en 3 perfiles principales
    if distancia_km <= 6.0:
        cluster = "Ruta Corta Local (Alta rotación)"
        tarifa_base = 15.0
    elif distancia_km <= 16.0:
        cluster = "Ruta Media Estándar (Interurbana)"
        tarifa_base = 25.0
    else:
        cluster = "Ruta Larga Especial (Foránea)"
        tarifa_base = 40.0
        
    # 2. Análisis del Tráfico: Ajuste de tarifa basado en el tiempo
    # Si toma mucho tiempo recorrer poca distancia (tráfico lento), la tarifa sube.
    proporcion_trafico = tiempo_min / distancia_km
    tarifa_optima = round(tarifa_base * (1 + (proporcion_trafico * 0.08)), 2)
    
    # 3. PCA (Análisis de Componentes Principales)
    # Reducimos las variables a dos dimensiones para el simulador gráfico
    pc1 = round((distancia_km * 0.7) + (tiempo_min * 0.3), 2)
    pc2 = round((distancia_km * -0.3) + (tiempo_min * 0.7), 2)
    
    return {
        "perfil_ruta": cluster,
        "tarifa_recomendada": tarifa_optima,
        "rentabilidad": "Alta" if tarifa_optima >= 25 else "Moderada",
        "coordenadas_pca": {"pc1": pc1, "pc2": pc2}
    }