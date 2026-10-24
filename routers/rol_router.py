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
    """Ruta HTTP GET para mostrar la vista principal con el listado de todos los roles."""
    # Obtenemos todos los registros de roles utilizando la clase RolController
    lista_roles = RolController.obtener_todos_los_roles()

    # Renderizamos y retornamos la plantilla HTML 'rol_inicio.html' enviando los roles obtenidos
    return motor_plantillas.TemplateResponse(
        solicitud,
        "rol/rol_inicio.html",
        {
            "roles": lista_roles,
        },
    )


@router.get("/nuevo")
def vista_crear_rol(solicitud: Request):
    """Ruta HTTP GET para cargar el formulario vacío destinado a registrar un nuevo rol."""
    # Renderizamos la plantilla de formulario pasando 'rol' como None para indicar que es un registro nuevo
    return motor_plantillas.TemplateResponse(
        solicitud, "rol/rol_form.html", {"rol": None}
    )


@router.post("/guardar")
def guardar_rol(nombre_rol: str = Form(...)):
    """Ruta HTTP POST para procesar el formulario de creación y almacenar el nuevo rol."""
    # Llamamos al método de la clase controlador para crear e insertar el rol en la base de datos
    RolController.crear_rol(nombre_rol)

    # Redirigimos al usuario a la vista principal de roles utilizando el código HTTP 303 (See Other)
    return RedirectResponse(
        url="/rol",
        status_code=status.HTTP_303_SEE_OTHER,
    )


@router.get("/editar/{id_rol}")
def vista_editar_rol(solicitud: Request, id_rol: int):
    """Ruta HTTP GET para cargar los datos de un rol existente dentro del formulario de edición."""
    # Buscamos el rol utilizando nuestro controlador basado en clases
    rol_encontrado = RolController.obtener_rol_por_id(id_rol)

    # Renderizamos la plantilla del formulario enviando los datos del rol encontrado
    return motor_plantillas.TemplateResponse(
        solicitud, "rol/rol_form.html", {"rol": rol_encontrado}
    )


@router.post("/actualizar")
def actualizar_rol(id_rol: int = Form(...), nombre_rol: str = Form(...)):
    """Ruta HTTP POST para procesar la actualización del nombre de un rol existente."""
    # Delegamos a la clase controlador la actualización del registro en la base de datos
    RolController.actualizar_rol(id_rol, nombre_rol)

    # Redirigimos de vuelta al listado principal con el código de redirección correcto
    return RedirectResponse(
        url="/rol",
        status_code=status.HTTP_303_SEE_OTHER,
    )


@router.get("/eliminar/{id_rol}")
def eliminar_rol(id_rol: int):
    """Ruta HTTP GET para procesar la eliminación definitiva de un rol según su identificador."""
    # Solicitamos a la clase controlador que elimine el rol correspondiente
    RolController.eliminar_rol(id_rol)

    # Redirigimos al listado principal de roles una vez completada la baja
    return RedirectResponse(
        url="/rol",
        status_code=status.HTTP_303_SEE_OTHER,
    )