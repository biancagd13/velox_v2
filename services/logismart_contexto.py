# ============================================================
# VELOX UTVT - CONTEXTO ESTRUCTURADO PARA LOGISMART
# Une reglas, busqueda BFS, analitica Atlas y estimador
# ============================================================

from typing import Any

from services.analitica_atlas import (
    obtener_resumen_general_viajes,
    obtener_top_rutas_frecuentes,
)
from services.busqueda_rutas import obtener_ruta_velox
from services.estimador_ocupacion import estimar_ocupacion_viaje
from services.reglas_viaje import validar_viaje_para_reserva


def construir_contexto_logismart(
    origen: str | None = None,
    destino: str | None = None,
    correo_usuario: str | None = None,
    viaje_candidato: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Arma un diccionario con datos reales y calculados de Velox UTVT
    para que LogiSmart responda con base en reglas, rutas y metricas.
    """
    resumen_bd = obtener_resumen_general_viajes()
    top_rutas = obtener_top_rutas_frecuentes(limite=3)

    info_ruta = None
    if origen and destino:
        info_ruta = obtener_ruta_velox(origen, destino)

    info_validacion = None
    info_ocupacion = None
    if viaje_candidato:
        costo = float(viaje_candidato.get("costo") or 0.0)
        asientos = int(float(viaje_candidato.get("asientos_disponibles") or 0))
        capacidad = viaje_candidato.get("capacidad_total") or viaje_candidato.get("asientos_totales")
        cap_int = int(float(capacidad)) if capacidad is not None else None

        info_ocupacion = estimar_ocupacion_viaje(
            costo=costo,
            asientos_disponibles=asientos,
            capacidad_total=cap_int,
        )

        if correo_usuario:
            es_valido, motivo = validar_viaje_para_reserva(
                viaje=viaje_candidato,
                correo_solicitante=correo_usuario,
            )
            info_validacion = {
                "es_valido": es_valido,
                "motivo": motivo,
            }

    return {
        "resumen_general": resumen_bd,
        "top_rutas": top_rutas,
        "ruta_consultada": info_ruta,
        "ocupacion_estimada": info_ocupacion,
        "validacion_reserva": info_validacion,
    }


def formatear_contexto_para_prompt(contexto: dict[str, Any]) -> str:
    """Convierte el contexto en un bloque legible para el prompt de LogiSmart."""
    lineas: list[str] = ["[CONTEXTO EN TIEMPO REAL - VELOX UTVT]"]

    resumen = contexto.get("resumen_general") or {}
    lineas.append(
        f"- Viajes registrados: {resumen.get('total_viajes', 0)} | "
        f"Costo promedio: ${resumen.get('costo_promedio', 0)} | "
        f"Asientos disponibles promedio: {resumen.get('asientos_promedio', 0)}"
    )

    top_rutas = contexto.get("top_rutas") or []
    if top_rutas:
        rutas_txt = ", ".join(
            f"{r.get('origen')} -> {r.get('destino')} ({r.get('total_viajes')} viajes, prom ${r.get('costo_promedio')})"
            for r in top_rutas
        )
        lineas.append(f"- Rutas mas frecuentes: {rutas_txt}")

    ruta = contexto.get("ruta_consultada")
    if ruta:
        if ruta.get("encontrada"):
            trayecto = " -> ".join(ruta.get("ruta") or [])
            lineas.append(f"- Ruta BFS ({ruta.get('conexiones')} conexiones): {trayecto}")
        else:
            lineas.append(f"- Ruta BFS: {ruta.get('mensaje')}")

    ocup = contexto.get("ocupacion_estimada")
    if ocup:
        lineas.append(
            f"- Estimacion de llenado: {ocup.get('porcentaje_llenado_estimado')}% "
            f"(demanda {ocup.get('nivel_demanda')}, ~{ocup.get('asientos_ocupados_estimados')} ocupados)"
        )

    val = contexto.get("validacion_reserva")
    if val:
        estado_txt = "APROBADA POR REGLAS" if val.get("es_valido") else "BLOQUEADA POR REGLAS"
        lineas.append(f"- Validacion de reserva: {estado_txt} ({val.get('motivo')})")

    return "\n".join(lineas)


if __name__ == "__main__":
    ctx = construir_contexto_logismart(
        origen="Metepec Centro",
        destino="UTVT",
        correo_usuario="alumno@utvt.edu.mx",
        viaje_candidato={
            "id_viaje": "VJ-200",
            "correo_conductor": "chofer@utvt.edu.mx",
            "asientos_disponibles": 2,
            "capacidad_total": 4,
            "costo": 30.0,
            "estado": "disponible",
        },
    )
    print(formatear_contexto_para_prompt(ctx))