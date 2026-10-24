# ==========================================================
# ARCHIVO PRINCIPAL DEL SERVIDOR: Velox UTVT API
# ==========================================================

# Importación del módulo del sistema para acceder a variables y funciones del intérprete de Python
import sys

# Importación de la clase Path del módulo pathlib para manejar rutas de archivos y directorios de forma multiplataforma
from pathlib import Path

# Agregamos dinámicamente el directorio raíz actual del proyecto al PATH de Python para prevenir errores de importación de paquetes locales
sys.path.append(str(Path(__file__).resolve().parent))

# Importación del framework FastAPI para inicializar la aplicación web principal y el servidor de API
from fastapi import FastAPI

# Importación de RedirectResponse para gestionar redirecciones HTTP automáticas y seguras entre endpoints
from fastapi.responses import RedirectResponse

# Importación para servir archivos estáticos (CSS, JS, imágenes)
from fastapi.staticfiles import StaticFiles

# Importación formal de todos los enrutadores (routers) correspondientes a cada módulo del sistema Velox UTVT
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


# ==========================================================
# INICIALIZACIÓN DE LA APLICACIÓN
# ==========================================================
app = FastAPI(
    title="Velox UTVT API", 
    description="Sistema institucional de carpooling para la Universidad Tecnológica del Valle de Toluca",
    version="1.0"
)


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

# Montaje de la carpeta estática para estilos CSS y recursos visuales
app.mount("/static", StaticFiles(directory="static"), name="static")


# ==========================================================
# RUTA RAÍZ
# ==========================================================
@app.get("/")
def raiz():
    """
    Ruta raíz del servidor: Redirige automáticamente la petición HTTP del usuario 
    hacia el primer módulo principal de gestión del sistema (Gestión de Roles).
    """
    return RedirectResponse(url="/rol", status_code=303)


# ==========================================================
# EJECUCIÓN DEL SERVIDOR LOCAL
# ==========================================================
if __name__ == "__main__":
    # Importación condicional del servidor ASGI Uvicorn optimizado para desarrollo local
    import uvicorn

    # Ejecución de la aplicación apuntando al módulo e instancia actual ('main:app'), 
    # configurando la dirección IP local (host), el puerto de escucha (8000) y el autoreinicio en caliente (reload)
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)