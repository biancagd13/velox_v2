# ==============================================================================
# ENRUTADOR DEDICADO PARA LOGISMART - VELOX UTVT
# Nuevo router sin modificar routers/asistente_router.py
# ==============================================================================

from typing import Any

from fastapi import APIRouter
from pydantic import BaseModel

from services.logismart_contexto import (
    construir_contexto_logismart,
    formatear_contexto_para_prompt,
)

router_logismart = APIRouter(
    prefix="/logismart",
    tags=["LogiSmart - Asistente Velox UTVT"],
)


class PeticionAsistenteLogiSmart(BaseModel):
    mensaje: str
    origen: str | None = None
    destino: str | None = None
    correo_usuario: str | None = None
    viaje_candidato: dict[str, Any] | None = None


@router_logismart.post("/preparar-respuesta")
def preparar_respuesta_logismart(payload: PeticionAsistenteLogiSmart):
    """
    Construye el contexto de datos reales, reglas, ruta BFS y estimacion
    de ocupacion para que LogiSmart responda con base en informacion actual.
    """
    contexto = construir_contexto_logismart(
        origen=payload.origen,
        destino=payload.destino,
        correo_usuario=payload.correo_usuario,
        viaje_candidato=payload.viaje_candidato,
    )
    bloque_prompt = formatear_contexto_para_prompt(contexto)

    prompt_enriquecido = (
        f"{bloque_prompt}\n\n"
        f"[MENSAJE DEL USUARIO]\n{payload.mensaje.strip()}"
    )

    return {
        "mensaje_usuario": payload.mensaje.strip(),
        "contexto": contexto,
        "bloque_contexto": bloque_prompt,
        "prompt_enriquecido": prompt_enriquecido,
    }