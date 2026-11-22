# ==============================================================================
# ENRUTADOR DE SERVICIOS DE IA Y LOGISMART - VELOX UTVT
# ==============================================================================

import os
import sys
from typing import Any

from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field

RAIZ_PROYECTO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if RAIZ_PROYECTO not in sys.path:
    sys.path.append(RAIZ_PROYECTO)

try:
    from database import db
except ImportError:
    db = None

from services.analitica_atlas import (
    obtener_resumen_general_viajes,
    obtener_top_rutas_frecuentes,
    obtener_viajes_por_estado,
)
from services.busqueda_rutas import obtener_ruta_velox
from services.calidad_viajes import auditar_viajes_bd
from services.estimador_ocupacion import estimar_ocupacion_viaje
from services.logismart_contexto import (
    construir_contexto_logismart,
    formatear_contexto_para_prompt,
)
from services.reglas_viaje import validar_viaje_para_reserva

router_ia_servicios = APIRouter(prefix="/ia/servicios", tags=["IA - Servicios Velox y LogiSmart"])
templates = Jinja2Templates(directory="templates")


class SolicitudValidacionReserva(BaseModel):
    correo_solicitante: str
    viaje: dict[str, Any]


class SolicitudEstimacionOcupacion(BaseModel):
    costo: float = Field(ge=0.0, le=500.0)
    asientos_disponibles: int = Field(ge=0, le=6)
    capacidad_total: int | None = Field(default=None, ge=1, le=10)


class SolicitudContextoLogiSmart(BaseModel):
    origen: str | None = None
    destino: str | None = None
    correo_usuario: str | None = None
    viaje_candidato: dict[str, Any] | None = None


def _obtener_viaje_real_inicial() -> dict[str, Any]:
    """
    Consulta el primer viaje real en db['viaje'], lo cruza con db['ruta'] y db['usuario']
    y devuelve un diccionario listo para inicializar el panel con datos reales.
    """
    respaldo = {
        "id_viaje": "",
        "origen": "Toluca",
        "destino": "UTVT",
        "costo": 15.0,
        "asientos_disponibles": 2,
        "capacidad_total": 4,
        "estado": "pendiente",
        "correo_conductor": "",
        "correo_usuario_ejemplo": "alumno@utvt.edu.mx",
    }

    if db is None:
        return respaldo

    try:
        viaje_doc = db["viaje"].find_one({}, {"_id": 0})
        if not viaje_doc:
            return respaldo

        ruta_doc = {}
        id_ruta_raw = viaje_doc.get("id_ruta")
        if id_ruta_raw is not None:
            ruta_doc = db["ruta"].find_one({"id_ruta": id_ruta_raw}, {"_id": 0}) or {}
            if not ruta_doc and str(id_ruta_raw).isdigit():
                ruta_doc = db["ruta"].find_one({"id_ruta": int(id_ruta_raw)}, {"_id": 0}) or {}
            if not ruta_doc:
                ruta_doc = db["ruta"].find_one({"id_ruta": str(id_ruta_raw)}, {"_id": 0}) or {}

        usuario_doc = {}
        id_cond_raw = viaje_doc.get("id_usuario") or viaje_doc.get("id_conductor") or ruta_doc.get("id_usuario")
        if id_cond_raw is not None:
            usuario_doc = db["usuario"].find_one({"id_usuario": id_cond_raw}, {"_id": 0}) or {}
            if not usuario_doc and str(id_cond_raw).isdigit():
                usuario_doc = db["usuario"].find_one({"id_usuario": int(id_cond_raw)}, {"_id": 0}) or {}

        origen = str(viaje_doc.get("origen") or ruta_doc.get("origen") or respaldo["origen"]).strip()
        destino = str(viaje_doc.get("destino") or ruta_doc.get("destino") or respaldo["destino"]).strip()

        costo_raw = viaje_doc.get("costo") if viaje_doc.get("costo") is not None else ruta_doc.get("costo", respaldo["costo"])
        costo = max(0.0, min(500.0, float(costo_raw or 0.0)))

        asientos_raw = (
            viaje_doc.get("asientos_disponibles")
            if viaje_doc.get("asientos_disponibles") is not None
            else ruta_doc.get("lugares", respaldo["asientos_disponibles"])
        )
        asientos = max(0, min(6, int(float(asientos_raw or 0))))

        capacidad_raw = viaje_doc.get("capacidad_total") or viaje_doc.get("capacidad_lugares") or 4
        capacidad = max(max(1, asientos), min(10, int(float(capacidad_raw or 4))))

        estado = str(viaje_doc.get("estado") or respaldo["estado"]).strip().lower()
        correo_cond = str(
            usuario_doc.get("correo_institucional") or usuario_doc.get("correo") or ""
        ).strip()

        return {
            "id_viaje": str(viaje_doc.get("id_viaje") or ""),
            "origen": origen,
            "destino": destino,
            "costo": costo,
            "asientos_disponibles": asientos,
            "capacidad_total": capacidad,
            "estado": estado,
            "correo_conductor": correo_cond,
            "correo_usuario_ejemplo": "alumno@utvt.edu.mx",
        }
    except Exception as error:
        print(f"[Aviso Panel IA] No se pudo cargar viaje real de BD: {error}")
        return respaldo


@router_ia_servicios.get("/panel", response_class=HTMLResponse)
def vista_panel_servicios(request: Request):
    """Renderiza la vista web con datos reales de MongoDB Atlas, BFS, ocupación y LogiSmart."""
    resumen = obtener_resumen_general_viajes()
    top_rutas = obtener_top_rutas_frecuentes(limite=5)
    calidad = auditar_viajes_bd(limite=500)

    viaje_real = _obtener_viaje_real_inicial()
    ruta_ejemplo = obtener_ruta_velox(viaje_real["origen"], viaje_real["destino"])
    ocupacion_ejemplo = estimar_ocupacion_viaje(
        costo=viaje_real["costo"],
        asientos_disponibles=viaje_real["asientos_disponibles"],
        capacidad_total=viaje_real["capacidad_total"],
    )

    viaje_candidato = {
        "id_viaje": viaje_real["id_viaje"],
        "correo_conductor": viaje_real["correo_conductor"],
        "asientos_disponibles": viaje_real["asientos_disponibles"],
        "capacidad_total": viaje_real["capacidad_total"],
        "costo": viaje_real["costo"],
        "estado": viaje_real["estado"],
    }

    contexto = construir_contexto_logismart(
        origen=viaje_real["origen"],
        destino=viaje_real["destino"],
        correo_usuario=viaje_real["correo_usuario_ejemplo"],
        viaje_candidato=viaje_candidato,
    )
    bloque_logismart = formatear_contexto_para_prompt(contexto)

    return templates.TemplateResponse(
        request=request,
        name="ia_modulo/servicios_logismart.html",
        context={
            "resumen": resumen,
            "top_rutas": top_rutas,
            "calidad": calidad,
            "viaje_real": viaje_real,
            "ruta_ejemplo": ruta_ejemplo,
            "ocupacion_ejemplo": ocupacion_ejemplo,
            "bloque_logismart": bloque_logismart,
        },
    )


@router_ia_servicios.get("/ruta")
def consultar_ruta_bfs(origen: str, destino: str):
    """Busca conexion entre zonas mediante BFS usando BD y grafo base."""
    return obtener_ruta_velox(origen=origen, destino=destino)


@router_ia_servicios.post("/validar-reserva")
def validar_reserva(payload: SolicitudValidacionReserva):
    """Aplica reglas fijas deterministas antes de reservar un viaje."""
    es_valido, motivo = validar_viaje_para_reserva(
        viaje=payload.viaje,
        correo_solicitante=payload.correo_solicitante,
    )
    return {
        "es_valido": es_valido,
        "motivo": motivo,
    }


@router_ia_servicios.get("/calidad-viajes")
def calidad_viajes(limite: int = 500):
    """Audita registros de db.viaje y separa validos de los que tienen observaciones."""
    resumen = auditar_viajes_bd(limite=limite)
    return {
        "total": resumen.get("total", 0),
        "cantidad_validos": resumen.get("cantidad_validos", 0),
        "cantidad_observaciones": resumen.get("cantidad_observaciones", 0),
        "con_observaciones": resumen.get("con_observaciones", []),
        "aviso": resumen.get("aviso"),
    }


@router_ia_servicios.get("/analitica-atlas")
def analitica_atlas(limite_rutas: int = 5):
    """Ejecuta agregaciones en MongoDB Atlas sobre db.viaje."""
    return {
        "resumen_general": obtener_resumen_general_viajes(),
        "por_estado": obtener_viajes_por_estado(),
        "top_rutas": obtener_top_rutas_frecuentes(limite=limite_rutas),
    }


@router_ia_servicios.post("/estimar-ocupacion")
def estimar_ocupacion(payload: SolicitudEstimacionOcupacion):
    """Estima ocupacion y porcentaje de llenado de un viaje."""
    return estimar_ocupacion_viaje(
        costo=payload.costo,
        asientos_disponibles=payload.asientos_disponibles,
        capacidad_total=payload.capacidad_total,
    )


@router_ia_servicios.post("/logismart/contexto")
def contexto_logismart(payload: SolicitudContextoLogiSmart):
    """Devuelve contexto estructurado y bloque de texto listo para LogiSmart."""
    contexto = construir_contexto_logismart(
        origen=payload.origen,
        destino=payload.destino,
        correo_usuario=payload.correo_usuario,
        viaje_candidato=payload.viaje_candidato,
    )
    return {
        "contexto": contexto,
        "prompt_bloque": formatear_contexto_para_prompt(contexto),
    }