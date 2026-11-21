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
from routers.pasajero_router import router as pasajero_router
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
app.include_router(pasajero_router)
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


def _calcular_calificacion_conductor(id_conductor: int = 1) -> tuple[str, int]:
    """Obtiene el promedio en tiempo real del conductor desde MongoDB ('calificacion')."""
    try:
        docs = list(db["calificacion"].find({}, {"_id": 0}))
        valores = []
        for doc in docs:
            if "id_conductor" in doc and doc["id_conductor"] != id_conductor:
                continue
            val = doc.get("estrellas") or doc.get("puntuacion") or doc.get("calificacion")
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
    return "4.8", 0


# ==========================================================
# RUTA RAÍZ
# ==========================================================
@app.get("/")
def raiz():
    return RedirectResponse(url="/login", status_code=303)


# ==========================================================
# RUTAS DE PERFIL DE USUARIO
# ==========================================================
@app.get("/perfil")
async def ver_perfil(request: Request):
    total_usuarios = db["usuario"].count_documents({})
    total_viajes = db["viaje"].count_documents({})
    total_vehiculos = db["vehiculo"].count_documents({})

    usuario_actual = {
        "matricula": "222310001",
        "nombre": "Brandon",
        "apellido_paterno": "Rivera",
        "correo_institucional": "al222@utvt.edu.mx",
        "telefono": "7221234567",
        "rol": "Administrador",
        "calificacion": "4.9",
        "viajes_totales": total_viajes if total_viajes > 0 else 15,
        "vehiculos_registrados": total_vehiculos if total_vehiculos > 0 else 1,
        "usuarios_registrados": total_usuarios,
    }

    return templates.TemplateResponse(
        request=request,
        name="perfiles/perfil.html",
        context={"usuario": usuario_actual},
    )


@app.get("/perfil-conductor")
async def ver_perfil_conductor(request: Request):
    promedio_real, total_resenas = _calcular_calificacion_conductor(1)

    conductor_actual = {
        "matricula": "222310088",
        "nombre": "Brandon",
        "apellido_paterno": "Rivera",
        "correo_institucional": "al22288@utvt.edu.mx",
        "telefono": "7229876543",
        "rol": "Conductor",
        "calificacion": promedio_real,
        "viajes_completados": 34 + total_resenas,
        "ganancias_estimadas": 1700 + (total_resenas * 25),
        "vehiculo_principal": "Chevrolet Aveo 2023",
        "placas": "ABC-123-A",
        "color_vehiculo": "Blanco Platino",
    }

    return templates.TemplateResponse(
        request=request,
        name="perfiles/perfil_conductor.html",
        context={"usuario": conductor_actual},
    )


@app.get("/perfil-pasajero")
async def ver_perfil_pasajero():
    return RedirectResponse(url="/pasajero/", status_code=303)


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