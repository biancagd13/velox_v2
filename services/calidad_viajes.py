# ============================================================
# VELOX UTVT - CALIDAD Y LIMPIEZA DE REGISTROS DE VIAJES
# Conectado a MongoDB (db.viaje + db.ruta)
# ============================================================

import os
import sys
from typing import Any

import pandas as pd

RAIZ_PROYECTO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if RAIZ_PROYECTO not in sys.path:
    sys.path.append(RAIZ_PROYECTO)

try:
    from database import db
except ImportError:
    db = None

COSTO_MIN_VALIDO = 0.0
COSTO_MAX_VALIDO = 500.0
ASIENTOS_MIN_VALIDOS = 0
ASIENTOS_MAX_VALIDOS = 6
ESTADOS_VALIDOS = {
    "disponible",
    "pendiente",
    "activo",
    "programado",
    "en_curso",
    "iniciado",
    "comenzado",
    "finalizado",
    "finalizada",
    "completado",
    "cancelado",
    "cancelada",
}


def _obtener_mapa_rutas() -> dict[int, dict[str, Any]]:
    """Obtiene un diccionario {id_ruta: doc_ruta} desde db.ruta."""
    if db is None:
        return {}
    mapa: dict[int, dict[str, Any]] = {}
    try:
        for r in db["ruta"].find({}, {"_id": 0}):
            try:
                if r.get("id_ruta") is not None:
                    mapa[int(r["id_ruta"])] = r
            except (TypeError, ValueError):
                continue
    except Exception:
        pass
    return mapa


def limpiar_registro_viaje(
    doc: dict[str, Any],
    rutas_map: dict[int, dict[str, Any]] | None = None,
) -> tuple[bool, dict[str, Any], list[str]]:
    """
    Valida y normaliza un documento de viaje enriquecido con su ruta en db.ruta.
    Retorna (es_valido, registro_limpio, motivos_observacion).
    """
    observaciones: list[str] = []
    mapa_rutas = rutas_map if rutas_map is not None else _obtener_mapa_rutas()

    id_ruta_raw = doc.get("id_ruta")
    id_ruta_num: int | None = None
    try:
        if id_ruta_raw is not None and str(id_ruta_raw).strip() != "":
            id_ruta_num = int(id_ruta_raw)
    except (TypeError, ValueError):
        observaciones.append(f"id_ruta '{id_ruta_raw}' no numerico.")

    ruta_doc = mapa_rutas.get(id_ruta_num, {}) if id_ruta_num is not None else {}

    origen_raw = str(doc.get("origen") or ruta_doc.get("origen") or "").strip()
    destino_raw = str(doc.get("destino") or ruta_doc.get("destino") or "").strip()
    origen = " ".join(origen_raw.split()).title()
    destino = " ".join(destino_raw.split()).title()

    if not origen:
        observaciones.append("Origen vacio o faltante (revisar ruta asociada).")
    if not destino:
        observaciones.append("Destino vacio o faltante (revisar ruta asociada).")
    if origen and destino and origen == destino:
        observaciones.append("Origen y destino son iguales.")

    try:
        costo_raw = doc.get("costo") if doc.get("costo") is not None else ruta_doc.get("costo", 0.0)
        costo = float(costo_raw)
    except (TypeError, ValueError):
        costo = 0.0
        observaciones.append("Costo no numerico; se asigno 0.0.")

    if costo < COSTO_MIN_VALIDO or costo > COSTO_MAX_VALIDO:
        observaciones.append(f"Costo ({costo}) fuera del rango [{COSTO_MIN_VALIDO}, {COSTO_MAX_VALIDO}].")

    try:
        asientos_raw = (
            doc.get("asientos_disponibles")
            if doc.get("asientos_disponibles") is not None
            else ruta_doc.get("lugares", 0)
        )
        asientos = int(float(asientos_raw))
    except (TypeError, ValueError):
        asientos = 0
        observaciones.append("Asientos no numericos; se asigno 0.")

    if asientos < ASIENTOS_MIN_VALIDOS or asientos > ASIENTOS_MAX_VALIDOS:
        observaciones.append(
            f"Asientos ({asientos}) fuera del rango [{ASIENTOS_MIN_VALIDOS}, {ASIENTOS_MAX_VALIDOS}]."
        )

    estado = str(doc.get("estado") or "pendiente").strip().lower()
    if estado not in ESTADOS_VALIDOS:
        observaciones.append(f"Estado '{estado}' no reconocido.")

    registro_limpio = {
        "id_viaje": str(doc.get("id_viaje") or doc.get("_id") or "").strip(),
        "id_ruta": id_ruta_num if id_ruta_num is not None else 0,
        "origen": origen,
        "destino": destino,
        "costo": round(max(COSTO_MIN_VALIDO, min(COSTO_MAX_VALIDO, costo)), 2),
        "asientos_disponibles": max(0, min(ASIENTOS_MAX_VALIDOS, asientos)),
        "estado": estado,
    }

    es_valido = len(observaciones) == 0
    return es_valido, registro_limpio, observaciones


def auditar_viajes_bd(limite: int = 500) -> dict[str, Any]:
    """
    Lee viajes desde db.viaje, los cruza con db.ruta por id_ruta,
    separa registros validos vs. con observaciones y devuelve el resumen.
    """
    validos: list[dict[str, Any]] = []
    con_observaciones: list[dict[str, Any]] = []

    if db is None:
        return {
            "total": 0,
            "cantidad_validos": 0,
            "cantidad_observaciones": 0,
            "validos": [],
            "con_observaciones": [],
            "df_validos": pd.DataFrame(),
            "aviso": "Conexion a base de datos no disponible.",
        }

    try:
        rutas_map = _obtener_mapa_rutas()
        cursor = db["viaje"].find({}).limit(limite)
        for doc in cursor:
            es_valido, limpio, motivos = limpiar_registro_viaje(doc, rutas_map=rutas_map)
            if es_valido:
                validos.append(limpio)
            else:
                con_observaciones.append({
                    **limpio,
                    "motivos": motivos,
                })
    except Exception as error:
        print(f"[Aviso Calidad BD] Error auditando viajes: {error}")

    total = len(validos) + len(con_observaciones)
    return {
        "total": total,
        "cantidad_validos": len(validos),
        "cantidad_observaciones": len(con_observaciones),
        "validos": validos,
        "con_observaciones": con_observaciones,
        "df_validos": pd.DataFrame(validos),
    }