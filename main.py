# ==========================================================
# ARCHIVO PRINCIPAL DEL SERVIDOR: Velox UTVT API
# ==========================================================

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent))

from fastapi import FastAPI, Request
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from database import db
from routers.rol_router import router as rol_router
from routers.usuario_router import router as usuario_router
from routers.vehiculo_router import router as vehiculo_router
from routers.viaje_router import router as viaje_router
from routers.reserva_router import router as reserva_router
from routers.ruta_router import router as ruta_router
from routers.calificacion_router import router as calificacion_router
from routers import asistente_router
from routers import estadisticas_router
from routers import calificaciones_estadisticas_router
from routers import simulador_ia_router
from routers import ia_modelo_router
from routers import practica04_router
from routers import practica05_router
from routers import practica06_router
from routers import practica07_router
from routers import practica08_router
from routers import practica09_router


# ==========================================================
# INICIALIZACIÓN DE LA APLICACIÓN Y PLANTILLAS
# ==========================================================
app = FastAPI(
    title="Velox UTVT API",
    description="Sistema institucional de carpooling para la Universidad Tecnológica del Valle de Toluca",
    version="1.0",
)

templates = Jinja2Templates(directory="templates")


# ==========================================================
# REGISTRO DE ENRUTADORES (ROUTERS) Y ARCHIVOS ESTÁTICOS
# ==========================================================
app.include_router(rol_router)
app.include_router(usuario_router)
app.include_router(vehiculo_router)
app.include_router(viaje_router)
app.include_router(reserva_router)
app.include_router(ruta_router)
app.include_router(calificacion_router)
app.include_router(asistente_router.router_asistente)
app.include_router(estadisticas_router.router_estadisticas)
app.include_router(calificaciones_estadisticas_router.router_calificaciones_estadisticas)
app.include_router(simulador_ia_router.router_simulador_ia)
app.include_router(ia_modelo_router.router_ia_modelo)
app.include_router(practica04_router.router)
app.include_router(practica05_router.router)
app.include_router(practica06_router.router)
app.include_router(practica07_router.router)
app.include_router(practica08_router.router)
app.include_router(practica09_router.router)

app.mount("/static", StaticFiles(directory="static"), name="static")


def _resolver_id_usuario(request: Request, rol_objetivo: str | None = None) -> int | None:
    """Obtiene el id_usuario desde query, header o cookie, o el primero en BD según rol si aplica."""
    raw = (
        request.query_params.get("id_usuario")
        or request.headers.get("x-velox-usuario-id")
        or request.cookies.get("velox_usuario_id")
        or ""
    ).strip()
    if raw.isdigit():
        return int(raw)

    filtro = {}
    if rol_objetivo:
        filtro["rol"] = {"$regex": rol_objetivo, "$options": "i"}

    doc = db["usuario"].find_one(filtro, {"_id": 0, "id_usuario": 1})
    if doc and doc.get("id_usuario") is not None:
        try:
            return int(doc["id_usuario"])
        except (TypeError, ValueError):
            return None
    return None


def _obtener_usuario_bd(id_usuario: int | None) -> dict:
    """Recupera el documento de usuario desde MongoDB sin _id."""
    if id_usuario is None:
        return {}
    doc = db["usuario"].find_one({"id_usuario": id_usuario}, {"_id": 0})
    return doc or {}


def _calcular_calificacion_usuario(id_usuario: int | None) -> tuple[str, int]:
    """Calcula el promedio de calificaciones y conteo desde MongoDB para el usuario indicado."""
    if id_usuario is None:
        return "0.0", 0
    try:
        docs = list(
            db["calificacion"].find(
                {
                    "$or": [
                        {"id_conductor": id_usuario},
                        {"id_usuario": id_usuario},
                        {"id_pasajero": id_usuario},
                    ]
                },
                {"_id": 0},
            )
        )
        valores = []
        for doc in docs:
            val = doc.get("estrellas") if doc.get("estrellas") is not None else (
                doc.get("puntuacion") if doc.get("puntuacion") is not None else doc.get("calificacion")
            )
            try:
                num = float(val)
                if 1.0 <= num <= 5.0:
                    valores.append(num)
            except (TypeError, ValueError):
                continue
        if valores:
            return f"{round(sum(valores) / len(valores), 1):.1f}", len(valores)
    except Exception:
        pass
    return "0.0", 0


# ==========================================================
# RUTA RAÍZ
# ==========================================================
@app.get("/")
def raiz():
    return RedirectResponse(url="/login", status_code=303)


# ==========================================================
# RUTAS DE PERFIL Y PASAJERO
# ==========================================================
@app.get("/perfil")
async def ver_perfil(request: Request):
    id_usuario = _resolver_id_usuario(request)
    usuario_bd = _obtener_usuario_bd(id_usuario)
    promedio_real, _ = _calcular_calificacion_usuario(id_usuario)

    total_usuarios = db["usuario"].count_documents({})
    total_viajes = db["viaje"].count_documents({})
    total_vehiculos = db["vehiculo"].count_documents({})

    usuario_actual = {
        **usuario_bd,
        "calificacion": usuario_bd.get("calificacion", promedio_real),
        "viajes_totales": total_viajes,
        "vehiculos_registrados": total_vehiculos,
        "usuarios_registrados": total_usuarios,
    }

    return templates.TemplateResponse(
        request=request,
        name="perfiles/perfil.html",
        context={"usuario": usuario_actual},
    )


@app.get("/perfil-conductor")
async def ver_perfil_conductor(request: Request):
    id_usuario = _resolver_id_usuario(request, rol_objetivo="conductor")
    usuario_bd = _obtener_usuario_bd(id_usuario)
    promedio_real, total_resenas = _calcular_calificacion_usuario(id_usuario)

    vehiculo_bd = {}
    if id_usuario is not None:
        vehiculo_bd = db["vehiculo"].find_one(
            {
                "$or": [
                    {"id_usuario": id_usuario},
                    {"id_conductor": id_usuario},
                    {"usuario_id": id_usuario},
                ]
            },
            {"_id": 0},
        ) or {}

    viajes_usuario = 0
    if id_usuario is not None:
        viajes_usuario = db["viaje"].count_documents(
            {
                "$or": [
                    {"id_usuario": id_usuario},
                    {"id_conductor": id_usuario},
                    {"usuario_id": id_usuario},
                ]
            }
        )

    marca = str(vehiculo_bd.get("marca") or "").strip()
    modelo = str(vehiculo_bd.get("modelo") or "").strip()
    anio = str(vehiculo_bd.get("anio") or vehiculo_bd.get("año") or "").strip()
    vehiculo_texto = " ".join(p for p in [marca, modelo, anio] if p)

    usuario_actual = {
        **usuario_bd,
        "calificacion": promedio_real if total_resenas > 0 else usuario_bd.get("calificacion", "0.0"),
        "viajes_completados": viajes_usuario,
        "ganancias_estimadas": usuario_bd.get("ganancias_estimadas", 0),
        "vehiculo_principal": vehiculo_texto,
        "placas": vehiculo_bd.get("placas", ""),
        "color_vehiculo": vehiculo_bd.get("color", ""),
    }

    return templates.TemplateResponse(
        request=request,
        name="perfiles/perfil_conductor.html",
        context={"usuario": usuario_actual},
    )


@app.get("/perfil-pasajero")
async def ver_perfil_pasajero(request: Request):
    id_usuario = _resolver_id_usuario(request, rol_objetivo="pasajero")
    usuario_bd = _obtener_usuario_bd(id_usuario)
    promedio_real, _ = _calcular_calificacion_usuario(id_usuario)

    reservas_count = 0
    if id_usuario is not None:
        reservas_count = db["reserva"].count_documents(
            {
                "$or": [
                    {"id_usuario": id_usuario},
                    {"id_pasajero": id_usuario},
                    {"usuario_id": id_usuario},
                ]
            }
        )

    usuario_actual = {
        **usuario_bd,
        "calificacion": promedio_real,
        "viajes_completados": reservas_count,
        "ganancias_estimadas": usuario_bd.get("ganancias_estimadas", 0),
    }

    return templates.TemplateResponse(
        request=request,
        name="perfiles/perfil.html",
        context={"usuario": usuario_actual},
    )


@app.get("/pasajero/")
async def ver_buscar_viajes_pasajero(request: Request):
    id_usuario = _resolver_id_usuario(request, rol_objetivo="pasajero")
    usuario_bd = _obtener_usuario_bd(id_usuario)
    promedio_real, _ = _calcular_calificacion_usuario(id_usuario)

    reservas_count = 0
    if id_usuario is not None:
        reservas_count = db["reserva"].count_documents(
            {
                "$or": [
                    {"id_usuario": id_usuario},
                    {"id_pasajero": id_usuario},
                    {"usuario_id": id_usuario},
                ]
            }
        )

    usuario_actual = {
        **usuario_bd,
        "calificacion": promedio_real,
        "viajes_completados": reservas_count,
        "ganancias_estimadas": usuario_bd.get("ganancias_estimadas", 0),
    }

    return templates.TemplateResponse(
        request=request,
        name="pasajero/buscar_viajes.html",
        context={"usuario": usuario_actual},
    )


@app.get("/pasajero/mi-viaje")
async def ver_detalle_viaje_pasajero(request: Request):
    id_usuario = _resolver_id_usuario(request, rol_objetivo="pasajero")
    usuario_bd = _obtener_usuario_bd(id_usuario)
    promedio_real, _ = _calcular_calificacion_usuario(id_usuario)

    reservas_count = 0
    if id_usuario is not None:
        reservas_count = db["reserva"].count_documents(
            {
                "$or": [
                    {"id_usuario": id_usuario},
                    {"id_pasajero": id_usuario},
                    {"usuario_id": id_usuario},
                ]
            }
        )

    usuario_actual = {
        **usuario_bd,
        "calificacion": promedio_real,
        "viajes_completados": reservas_count,
        "ganancias_estimadas": usuario_bd.get("ganancias_estimadas", 0),
    }

    return templates.TemplateResponse(
        request=request,
        name="pasajero/detalle_viaje.html",
        context={"usuario": usuario_actual},
    )


@app.get("/pasajero/historial")
async def ver_historial_viajes_pasajero(request: Request):
    id_usuario = _resolver_id_usuario(request, rol_objetivo="pasajero")
    usuario_bd = _obtener_usuario_bd(id_usuario)
    promedio_real, _ = _calcular_calificacion_usuario(id_usuario)

    reservas_count = 0
    if id_usuario is not None:
        reservas_count = db["reserva"].count_documents(
            {
                "$or": [
                    {"id_usuario": id_usuario},
                    {"id_pasajero": id_usuario},
                    {"usuario_id": id_usuario},
                ]
            }
        )

    usuario_actual = {
        **usuario_bd,
        "calificacion": promedio_real,
        "viajes_completados": reservas_count,
        "ganancias_estimadas": usuario_bd.get("ganancias_estimadas", 0),
    }

    return templates.TemplateResponse(
        request=request,
        name="pasajero/historial_viajes.html",
        context={"usuario": usuario_actual},
    )


# ==========================================================
# RUTAS DE AUTENTICACIÓN (LOGIN / LOGOUT)
# ==========================================================
@app.get("/login")
async def vista_login(request: Request):
    return templates.TemplateResponse(
        request=request,
        name="sesion/login.html",
    )


@app.get("/logout")
async def cerrar_sesion():
    return RedirectResponse(url="/login", status_code=303)


# ==========================================================
# EJECUCIÓN DEL SERVIDOR LOCAL
# ==========================================================
if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="127.0.0.1", port=8080, reload=True)