# Importamos la clase controladora completa para gestionar la lógica de negocio de los vehículos
from controllers.vehiculo_controller import VehiculoController

# Importamos los modelos de Pydantic necesarios para validar los datos del vehículo
from models.vehiculo_model import VehiculoModel, VehiculoUpdateModel

# Importamos la base de datos para poder consultar los usuarios registrados
from database import db

# Importamos las herramientas de FastAPI: APIRouter para rutas, Form para datos de formularios y Request para solicitudes HTTP
from fastapi import APIRouter, Form, Request, status

# Importamos RedirectResponse para redirigir entre endpoints y Jinja2Templates para renderizar plantillas HTML
from fastapi.responses import RedirectResponse
from fastapi.templating import Jinja2Templates

# Inicializamos el enrutador con el prefijo y etiqueta correspondientes en español
router = APIRouter(prefix="/vehiculo", tags=["Vehículo"])

# Configuramos el directorio base donde se encuentran las plantillas HTML (carpeta 'templates')
motor_plantillas = Jinja2Templates(directory="templates")


def _es_admin(solicitud: Request) -> bool:
    """Detecta si la petición pertenece a una sesión de Administrador."""
    rol_query = (solicitud.query_params.get("rol") or "").strip().lower()
    rol_header = (solicitud.headers.get("x-velox-rol") or "").strip().lower()
    rol_cookie = (solicitud.cookies.get("velox_usuario_rol") or "").strip().lower()
    return "admin" in rol_query or "admin" in rol_header or "admin" in rol_cookie


@router.get("/")
def listar_vehiculos(solicitud: Request):
    """Ruta HTTP GET para mostrar la vista principal con el listado de todos los vehículos."""
    lista_vehiculos = VehiculoController.obtener_todos_los_vehiculos()
    es_admin = _es_admin(solicitud)
    plantilla = "admin/vehiculo_inicio_admin.html" if es_admin else "vehiculo/vehiculo_inicio.html"

    id_usuario_query = (solicitud.query_params.get("id_usuario") or "").strip()
    id_usuario_cookie = (solicitud.cookies.get("velox_usuario_id") or "").strip()
    id_usuario_activo = id_usuario_query or id_usuario_cookie

    vehiculo_actual = None
    if lista_vehiculos:
        if id_usuario_activo:
            vehiculo_actual = next(
                (v for v in lista_vehiculos if str(v.get("id_usuario", "")).strip() == id_usuario_activo),
                None,
            )
        if vehiculo_actual is None:
            vehiculo_actual = lista_vehiculos[0]

    return motor_plantillas.TemplateResponse(
        solicitud,
        plantilla,
        {
            "vehiculo": vehiculo_actual,
            "vehiculos": lista_vehiculos,
            "es_admin": es_admin,
        },
    )


@router.get("/nuevo")
def vista_crear_vehiculo(solicitud: Request):
    """Ruta HTTP GET para cargar el formulario vacío destinado a registrar un nuevo vehículo."""
    usuarios = list(db["usuario"].find({}, {"_id": 0}))
    es_admin = _es_admin(solicitud)
    plantilla = "admin/vehiculo_form_admin.html" if es_admin else "vehiculo/vehiculo_form.html"

    return motor_plantillas.TemplateResponse(
        solicitud,
        plantilla,
        {
            "vehiculo": None,
            "usuarios": usuarios,
            "es_admin": es_admin,
        },
    )


@router.post("/guardar")
def guardar_vehiculo(
    solicitud: Request,
    placa: str = Form(...),
    nombre: str = Form(...),
    modelo: str = Form(...),
    color: str = Form(...),
    capacidad_asientos: int = Form(...),
    id_usuario: int = Form(...),
):
    """Ruta HTTP POST para procesar el formulario de creación y almacenar el nuevo vehículo."""
    nuevo_vehiculo = VehiculoModel(
        placa=placa,
        nombre=nombre,
        modelo=modelo,
        color=color,
        capacidad_asientos=capacidad_asientos,
        id_usuario=id_usuario,
    )

    VehiculoController.crear_vehiculo(nuevo_vehiculo)

    url_retorno = "/vehiculo?rol=admin" if _es_admin(solicitud) else "/vehiculo"
    return RedirectResponse(
        url=url_retorno,
        status_code=status.HTTP_303_SEE_OTHER,
    )


@router.get("/editar/{id_vehiculo}")
def vista_editar_vehiculo(solicitud: Request, id_vehiculo: int):
    """Ruta HTTP GET para cargar los datos de un vehículo existente dentro del formulario de edición."""
    vehiculo_encontrado = VehiculoController.obtener_vehiculo_por_id(id_vehiculo)
    usuarios = list(db["usuario"].find({}, {"_id": 0}))
    es_admin = _es_admin(solicitud)
    plantilla = "admin/vehiculo_form_admin.html" if es_admin else "vehiculo/vehiculo_form.html"

    return motor_plantillas.TemplateResponse(
        solicitud,
        plantilla,
        {
            "vehiculo": vehiculo_encontrado,
            "usuarios": usuarios,
            "es_admin": es_admin,
        },
    )


@router.post("/actualizar")
def actualizar_vehiculo(
    solicitud: Request,
    id_vehiculo: int = Form(...),
    placa: str = Form(None),
    nombre: str = Form(None),
    modelo: str = Form(None),
    color: str = Form(None),
    capacidad_asientos: int = Form(None),
    id_usuario: int = Form(None),
):
    """Ruta HTTP POST para procesar la actualización parcial de un vehículo existente."""
    vehiculo_actualizacion = VehiculoUpdateModel(
        placa=placa,
        nombre=nombre,
        modelo=modelo,
        color=color,
        capacidad_asientos=capacidad_asientos,
        id_usuario=id_usuario,
    )

    VehiculoController.actualizar_vehiculo(id_vehiculo, vehiculo_actualizacion)

    url_retorno = "/vehiculo?rol=admin" if _es_admin(solicitud) else "/vehiculo"
    return RedirectResponse(
        url=url_retorno,
        status_code=status.HTTP_303_SEE_OTHER,
    )


@router.get("/eliminar/{id_vehiculo}")
def eliminar_vehiculo(solicitud: Request, id_vehiculo: int):
    """Ruta HTTP GET para procesar la eliminación definitiva de un vehículo."""
    VehiculoController.eliminar_vehiculo(id_vehiculo)

    url_retorno = "/vehiculo?rol=admin" if _es_admin(solicitud) else "/vehiculo"
    return RedirectResponse(
        url=url_retorno,
        status_code=status.HTTP_303_SEE_OTHER,
    )