# ==============================================================================
# CHAT PARALELO LOGISMART V2 - VELOX UTVT
# Motor propio basado en tus servicios, reglas, BFS, ocupación y MongoDB Atlas
# ==============================================================================

from typing import Any

from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

from routers.logismart_router import (
    PeticionAsistenteLogiSmart,
    preparar_respuesta_logismart,
)

router_logismart_v2 = APIRouter(
    prefix="/logismart-v2",
    tags=["LogiSmart v2 - Chat Comparativo Velox UTVT"],
)

templates = Jinja2Templates(directory="templates")


class PeticionChatLogiSmartV2(PeticionAsistenteLogiSmart):
    historial: list[dict[str, str]] | None = None


def _extraer_resumen(ctx: dict[str, Any]) -> dict[str, Any]:
    return (
        ctx.get("resumen_atlas")
        or ctx.get("resumen_general")
        or ctx.get("resumen")
        or {}
    )


def _extraer_ocupacion(ctx: dict[str, Any]) -> dict[str, Any]:
    return (
        ctx.get("estimacion_ocupacion")
        or ctx.get("ocupacion")
        or {}
    )


def _extraer_regla(ctx: dict[str, Any]) -> dict[str, Any]:
    return (
        ctx.get("validacion_reserva")
        or ctx.get("reglas_reserva")
        or {}
    )


def _generar_respuesta_propia_logismart(preparado: dict[str, Any]) -> str:
    """
    Genera la respuesta de LogiSmart v2 usando exclusivamente el contexto real
    construido por los servicios de Velox UTVT (Atlas, BFS, ocupación y reglas).
    """
    mensaje = (preparado.get("mensaje_usuario") or "").strip().lower()
    ctx = preparado.get("contexto") or {}

    resumen = _extraer_resumen(ctx)
    top_rutas = ctx.get("top_rutas") or []
    bfs = ctx.get("ruta_bfs") or {}
    ocup = _extraer_ocupacion(ctx)
    regla = _extraer_regla(ctx)

    total_viajes = resumen.get("total_viajes", 0)
    costo_prom = resumen.get("costo_promedio", 0)
    asientos_prom = resumen.get("asientos_promedio", 0)

    partes: list[str] = []

    if any(p in mensaje for p in ("hola", "buenas", "hey", "que tal", "qué tal")):
        partes.append(
            f"¡Hola! Soy LogiSmart v2 de Velox UTVT. Actualmente hay {total_viajes} viajes registrados "
            f"con un costo promedio de ${costo_prom} MXN y {asientos_prom} asientos disponibles en promedio."
        )
    else:
        partes.append(
            f"Según los datos actuales de Velox UTVT: {total_viajes} viajes en BD, "
            f"costo promedio de ${costo_prom} MXN y {asientos_prom} asientos disponibles en promedio."
        )

    if top_rutas and any(p in mensaje for p in ("ruta", "rutas", "frecuente", "top", "viaje", "viajes")):
        resumen_rutas = ", ".join(
            f"{r.get('origen', '?')} → {r.get('destino', '?')} ({r.get('total_viajes', 0)} viajes, prom ${r.get('costo_promedio', 0)} MXN)"
            for r in top_rutas[:3]
        )
        partes.append(f"Rutas frecuentes: {resumen_rutas}.")

    if bfs:
        if bfs.get("encontrada"):
            trayecto = " → ".join(bfs.get("ruta") or [])
            partes.append(
                f"Conexión BFS ({bfs.get('inicio')} → {bfs.get('destino')}): {trayecto} "
                f"({bfs.get('conexiones', 0)} tramos)."
            )
        elif bfs.get("mensaje") and any(p in mensaje for p in ("ruta", "bfs", "conexion", "conexión", "llegar", "ir")):
            partes.append(f"Búsqueda BFS ({bfs.get('inicio')} → {bfs.get('destino')}): {bfs.get('mensaje')}")

    if ocup and ocup.get("porcentaje_llenado_estimado") is not None:
        partes.append(
            f"Estimación de ocupación: {ocup.get('porcentaje_llenado_estimado')}% "
            f"(demanda {ocup.get('nivel_demanda', 'N/D')}, ~{ocup.get('asientos_ocupados_estimados', 0)} asientos ocupados)."
        )

    if regla and ("es_valido" in regla):
        estado_txt = "APROBADA" if regla.get("es_valido") else "BLOQUEADA"
        partes.append(f"Regla de reserva: {estado_txt} — {regla.get('motivo', '')}")

    return "\n".join(partes)


@router_logismart_v2.get("/", response_class=HTMLResponse)
def vista_chat_logismart_v2(request: Request):
    """Renderiza el chat paralelo LogiSmart v2 para comparar respuestas."""
    return templates.TemplateResponse(
        request=request,
        name="ia_modulo/logismart_v2.html",
        context={
            "modelo_ollama": "Motor Propio Velox UTVT",
        },
    )


@router_logismart_v2.post("/chat")
def enviar_mensaje_logismart_v2(payload: PeticionChatLogiSmartV2):
    """Prepara el contexto real con los servicios de Velox y responde con el motor propio."""
    preparado = preparar_respuesta_logismart(
        PeticionAsistenteLogiSmart(
            mensaje=payload.mensaje,
            origen=payload.origen,
            destino=payload.destino,
            correo_usuario=payload.correo_usuario,
            viaje_candidato=payload.viaje_candidato,
        )
    )

    respuesta = _generar_respuesta_propia_logismart(preparado)

    return JSONResponse(
        content={
            "ok": True,
            "respuesta": respuesta,
            "motor": "LogiSmart v2 · Motor Propio (MongoDB + BFS + Reglas)",
            "bloque_contexto": preparado["bloque_contexto"],
            "prompt_enriquecido": preparado["prompt_enriquecido"],
            "contexto": preparado["contexto"],
        }
    )