# ============================================================
# VELOX UTVT - REGLAS FIJAS DE VALIDACION DE VIAJES (BD)
# Conectado a MongoDB (database.db)
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

DOMINIOS_INSTITUCIONALES = (
    "@utvt.edu.mx",
    "@utvtol.edu.mx",
)

COSTO_MINIMO = 0.0
COSTO_MAXIMO = 500.0


def validar_correo_institucional(correo: str) -> tuple[bool, str]:
    """Verifica que el correo pertenezca a @utvt.edu.mx o @utvtol.edu.mx."""
    correo_limpio = (correo or "").strip().lower()
    if not correo_limpio.endswith(DOMINIOS_INSTITUCIONALES):
        return False, "El correo debe pertenecer a @utvt.edu.mx o @utvtol.edu.mx."
    return True, "Correo institucional valido."


def obtener_ids_viajes_activos_bd(correo_solicitante: str) -> set[str]:
    """Consulta en MongoDB si el usuario ya tiene solicitudes/reservas activas."""
    ids_activos: set[str] = set()
    if db is None or not correo_solicitante:
        return ids_activos

    correo_limpio = correo_solicitante.strip().lower()
    for nombre_col in ("reserva", "reservas", "solicitud", "solicitudes"):
        if nombre_col in db.list_collection_names():
            try:
                cursor = db[nombre_col].find(
                    {
                        "correo_usuario": correo_limpio,
                        "estado": {"$in": ["activa", "pendiente", "confirmada"]},
                    },
                    {"_id": 0, "id_viaje": 1, "viaje_id": 1},
                )
                for doc in cursor:
                    vid = doc.get("id_viaje") or doc.get("viaje_id")
                    if vid:
                        ids_activos.add(str(vid))
            except Exception as error:
                print(f"[Aviso Reglas BD] Error consultando {nombre_col}: {error}")
    return ids_activos


def validar_viaje_para_reserva(
    viaje: dict[str, Any],
    correo_solicitante: str,
    ids_viajes_activos_usuario: set[str] | None = None,
) -> tuple[bool, str]:
    """Aplica reglas fijas deterministas antes de permitir reservar un viaje."""
    correo_ok, motivo_correo = validar_correo_institucional(correo_solicitante)
    if not correo_ok:
        return False, motivo_correo

    estado_viaje = str(viaje.get("estado", "disponible")).strip().lower()
    if estado_viaje not in {"disponible", "activo", "programado", "en_curso"}:
        return False, f"El viaje no esta disponible (estado actual: {estado_viaje})."

    asientos_disponibles = int(viaje.get("asientos_disponibles", 0))
    if asientos_disponibles <= 0:
        return False, "No hay asientos disponibles en este viaje."

    costo = float(viaje.get("costo", 0.0))
    if costo < COSTO_MINIMO or costo > COSTO_MAXIMO:
        return False, f"El costo ({costo}) esta fuera del rango permitido."

    correo_conductor = str(viaje.get("correo_conductor", "")).strip().lower()
    correo_solicitante_limpio = correo_solicitante.strip().lower()
    if correo_conductor and correo_conductor == correo_solicitante_limpio:
        return False, "El conductor no puede reservar su propio viaje."

    activos = ids_viajes_activos_usuario
    if activos is None:
        activos = obtener_ids_viajes_activos_bd(correo_solicitante_limpio)

    id_viaje = str(viaje.get("id_viaje") or viaje.get("_id") or "").strip()
    if activos and id_viaje and id_viaje in activos:
        return False, "Ya cuentas con una reserva o solicitud activa en este viaje."

    return True, "Viaje valido para reserva."