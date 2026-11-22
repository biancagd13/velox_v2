# Importamos la clase controladora completa para gestionar la lógica de negocio de los roles
from controllers.rol_controller import RolController

# Importamos la conexión directa a la base de datos MongoDB (por si se usa de apoyo en consultas directas)
from database import db

# Importamos las herramientas de FastAPI: APIRouter para rutas, Form para datos de formularios y Request para solicitudes HTTP
from fastapi import APIRouter, Form, Request, status

# Importamos RedirectResponse para redirigir entre endpoints y Jinja2Templates para renderizar plantillas HTML
from fastapi.responses import RedirectResponse
from fastapi.templating import Jinja2Templates

# Inicializamos el enrutador con el nombre estándar 'router' (requerido por el main.py) y alias en español
router = APIRouter(prefix="/rol", tags=["Rol"])

# Configuramos el directorio base donde se encuentran las plantillas HTML (carpeta 'templates')
motor_plantillas = Jinja2Templates(directory="templates")


@router.get("/")
def listar_roles(solicitud: Request):
    """Ruta HTTP GET para mostrar la vista principal de roles (Administrador)."""
    lista_roles = RolController.obtener_todos_los_roles()

    return motor_plantillas.TemplateResponse(
        solicitud,
        "admin/rol_inicio_admin.html",
        {
            "roles": lista_roles,
        },
    )


@router.get("/nuevo")
def vista_crear_rol(solicitud: Request):
    """Ruta HTTP GET para cargar el formulario vacío destinado a registrar un nuevo rol."""
    return motor_plantillas.TemplateResponse(
        solicitud, "admin/rol_form_admin.html", {"rol": None}
    )


@router.post("/guardar")
def guardar_rol(nombre_rol: str = Form(...)):
    """Ruta HTTP POST para procesar el formulario de creación y almacenar el nuevo rol."""
    RolController.crear_rol(nombre_rol)

    return RedirectResponse(
        url="/rol",
        status_code=status.HTTP_303_SEE_OTHER,
    )


@router.get("/editar/{id_rol}")
def vista_editar_rol(solicitud: Request, id_rol: int):
    """Ruta HTTP GET para cargar los datos de un rol existente dentro del formulario de edición."""
    rol_encontrado = RolController.obtener_rol_por_id(id_rol)

    return motor_plantillas.TemplateResponse(
        solicitud, "admin/rol_form_admin.html", {"rol": rol_encontrado}
    )


@router.post("/actualizar")
def actualizar_rol(id_rol: int = Form(...), nombre_rol: str = Form(...)):
    """Ruta HTTP POST para procesar la actualización del nombre de un rol existente."""
    RolController.actualizar_rol(id_rol, nombre_rol)

    return RedirectResponse(
        url="/rol",
        status_code=status.HTTP_303_SEE_OTHER,
    )


@router.get("/eliminar/{id_rol}")
def eliminar_rol(id_rol: int):
    """Ruta HTTP GET para procesar la eliminación definitiva de un rol según su identificador."""
    RolController.eliminar_rol(id_rol)

    return RedirectResponse(
        url="/rol",
        status_code=status.HTTP_303_SEE_OTHER,
    )