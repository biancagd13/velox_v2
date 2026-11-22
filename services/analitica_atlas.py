# ============================================================
# VELOX UTVT - ANALITICA Y AGREGACIONES EN MONGODB ATLAS
# Basado en p16consultasinteligentes.txt y conectado a db.viaje
# ============================================================

import os
import sys
from typing import Any

RAIZ_PROYECTO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if RAIZ_PROYECTO not in sys.path:
    sys.path.append(RAIZ_PROYECTO)

try:
    from database import db
except ImportError:
    db = None


def obtener_resumen_general_viajes() -> dict[str, Any]:
    """Ejecuta un pipeline $group en MongoDB Atlas para obtener metricas clave."""
    resumen_vacio = {
        "total_viajes": 0,
        "costo_promedio": 0.0,
        "costo_minimo": 0.0,
        "costo_maximo": 0.0,
        "asientos_promedio": 0.0,
        "total_asientos_disponibles": 0,
    }

    if db is None:
        return resumen_vacio

    try:
        pipeline = [
            {
                "$group": {
                    "_id": None,
                    "total_viajes": {"$sum": 1},
                    "costo_promedio": {"$avg": {"$toDouble": {"$ifNull": ["$costo", 0]}}},
                    "costo_minimo": {"$min": {"$toDouble": {"$ifNull": ["$costo", 0]}}},
                    "costo_maximo": {"$max": {"$toDouble": {"$ifNull": ["$costo", 0]}}},
                    "asientos_promedio": {
                        "$avg": {"$toDouble": {"$ifNull": ["$asientos_disponibles", 0]}}
                    },
                    "total_asientos_disponibles": {
                        "$sum": {"$toInt": {"$ifNull": ["$asientos_disponibles", 0]}}
                    },
                }
            },
            {
                "$project": {
                    "_id": 0,
                    "total_viajes": 1,
                    "costo_promedio": {"$round": ["$costo_promedio", 2]},
                    "costo_minimo": {"$round": ["$costo_minimo", 2]},
                    "costo_maximo": {"$round": ["$costo_maximo", 2]},
                    "asientos_promedio": {"$round": ["$asientos_promedio", 2]},
                    "total_asientos_disponibles": 1,
                }
            },
        ]

        resultados = list(db.viaje.aggregate(pipeline))
        return resultados[0] if resultados else resumen_vacio
    except Exception as error:
        print(f"[Aviso Agregaciones Atlas] Error en resumen general: {error}")
        return resumen_vacio


def obtener_viajes_por_estado() -> list[dict[str, Any]]:
    """Agrupa viajes por estado y calcula su costo y asientos promedio."""
    if db is None:
        return []

    try:
        pipeline = [
            {
                "$group": {
                    "_id": {"$ifNull": ["$estado", "sin_estado"]},
                    "cantidad": {"$sum": 1},
                    "costo_promedio": {"$avg": {"$toDouble": {"$ifNull": ["$costo", 0]}}},
                    "asientos_promedio": {
                        "$avg": {"$toDouble": {"$ifNull": ["$asientos_disponibles", 0]}}
                    },
                }
            },
            {"$sort": {"cantidad": -1}},
            {
                "$project": {
                    "_id": 0,
                    "estado": "$_id",
                    "cantidad": 1,
                    "costo_promedio": {"$round": ["$costo_promedio", 2]},
                    "asientos_promedio": {"$round": ["$asientos_promedio", 2]},
                }
            },
        ]
        return list(db.viaje.aggregate(pipeline))
    except Exception as error:
        print(f"[Aviso Agregaciones Atlas] Error agrupando por estado: {error}")
        return []


def obtener_top_rutas_frecuentes(limite: int = 5) -> list[dict[str, Any]]:
    """Obtiene las rutas origen -> destino mas frecuentes uniendo viaje.id_ruta con ruta.id_ruta."""
    if db is None:
        return []

    try:
        pipeline = [
            {
                "$lookup": {
                    "from": "ruta",
                    "let": {"ruta_id_viaje": {"$toString": "$id_ruta"}},
                    "pipeline": [
                        {
                            "$match": {
                                "$expr": {
                                    "$eq": [{"$toString": "$id_ruta"}, "$$ruta_id_viaje"]
                                }
                            }
                        }
                    ],
                    "as": "ruta_info",
                }
            },
            {
                "$unwind": {
                    "path": "$ruta_info",
                    "preserveNullAndEmptyArrays": True,
                }
            },
            {
                "$addFields": {
                    "origen_resuelto": {"$ifNull": ["$origen", "$ruta_info.origen"]},
                    "destino_resuelto": {"$ifNull": ["$destino", "$ruta_info.destino"]},
                }
            },
            {
                "$match": {
                    "origen_resuelto": {"$nin": [None, ""]},
                    "destino_resuelto": {"$nin": [None, ""]},
                }
            },
            {
                "$group": {
                    "_id": {
                        "origen": "$origen_resuelto",
                        "destino": "$destino_resuelto",
                    },
                    "total_viajes": {"$sum": 1},
                    "costo_promedio": {"$avg": {"$toDouble": {"$ifNull": ["$costo", 0]}}},
                    "asientos_promedio": {
                        "$avg": {"$toDouble": {"$ifNull": ["$asientos_disponibles", 0]}}
                    },
                }
            },
            {"$sort": {"total_viajes": -1}},
            {"$limit": int(limite)},
            {
                "$project": {
                    "_id": 0,
                    "origen": "$_id.origen",
                    "destino": "$_id.destino",
                    "total_viajes": 1,
                    "costo_promedio": {"$round": ["$costo_promedio", 2]},
                    "asientos_promedio": {"$round": ["$asientos_promedio", 2]},
                }
            },
        ]
        return list(db["viaje"].aggregate(pipeline))
    except Exception as error:
        print(f"[Aviso Agregaciones Atlas] Error en top rutas: {error}")
        return []


if __name__ == "__main__":
    print("Resumen general:", obtener_resumen_general_viajes())
    print("Por estado:", obtener_viajes_por_estado())
    print("Top rutas:", obtener_top_rutas_frecuentes())