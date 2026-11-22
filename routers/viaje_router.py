# Importamos la clase controladora encargada de la lógica CRUD de los viajes
from controllers.viaje_controller import ViajeController

# Importamos los modelos de Pydantic para validar la creación y actualización de viajes
from models.viaje_model import ViajeModel, ViajeUpdateModel

# Importamos el enumerado oficial de estados para asignar el valor inicial consistente
from models.enums import EstadoViaje

# Importamos la conexión activa a la base de datos MongoDB
from database import db

# Importamos las herramientas de FastAPI para definir rutas, formularios y peticiones HTTP
from fastapi import APIRouter, Form, Request, status

# Importamos RedirectResponse para redirecciones seguras y Jinja2Templates para renderizar vistas HTML
from fastapi.responses import RedirectResponse
from fastapi.templating import Jinja2Templates

# Inicializamos el enrutador del módulo de Viajes con prefijo y etiqueta en español
router = APIRouter(prefix="/viaje", tags=["Viaje"])

# Configuramos el motor de plantillas Jinja2 apuntando al directorio 'templates'
motor_plantillas = Jinja2Templates(directory="templates")


def _es_admin(solicitud: Request) -> bool:
    """Detecta si la petición pertenece a una sesión de Administrador."""
    rol_query = (solicitud.query_params.get("rol") or "").strip().lower()
    rol_header = (solicitud.headers.get("x-velox-rol") or "").strip().lower()
    rol_cookie = (solicitud.cookies.get("velox_usuario_rol") or "").strip().lower()
    return "admin" in rol_query or "admin" in rol_header or "admin" in rol_cookie


def _obtener_id_conductor_actual() -> int:
    """Función auxiliar que obtiene el id_usuario del conductor autenticado desde MongoDB."""
    conductor = db["usuario"].find_one(
        {"correo_institucional": "al22288@utvt.edu.mx"},
        {"_id": 0, "id_usuario": 1},
    )
    if conductor and conductor.get("id_usuario") is not None:
        return int(conductor["id_usuario"])

    return 1


@router.get("/")
def listar_viajes(solicitud: Request):
    """Ruta HTTP GET para mostrar la vista principal con el listado de todos los viajes."""
    lista_viajes = ViajeController.obtener_todos_los_viajes()
    es_admin = _es_admin(solicitud)
    plantilla = "admin/viaje_inicio_admin.html" if es_admin else "viaje/viaje_inicio.html"

    return motor_plantillas.TemplateResponse(
        solicitud,
        plantilla,
        {
            "viajes": lista_viajes,
            "es_admin": es_admin,
        },
    )


@router.get("/nuevo")
def vista_crear_viaje(solicitud: Request):
    """Ruta HTTP GET para cargar el formulario vacío destinado a registrar un nuevo viaje."""
    id_conductor_actual = _obtener_id_conductor_actual()

    rutas_raw = list(db["ruta"].find({}, {"_id": 0}))
    rutas = []
    for r in rutas_raw:
        r["origen"] = r.get("origen", "Origen")
        r["destino"] = r.get("destino", "Destino")
        rutas.append(r)

    es_admin = _es_admin(solicitud)
    plantilla = "admin/viaje_form_admin.html" if es_admin else "viaje/viaje_form.html"

    return motor_plantillas.TemplateResponse(
        solicitud,
        plantilla,
        {
            "viaje": None,
            "rutas": rutas,
            "id_conductor_actual": id_conductor_actual,
            "es_admin": es_admin,
        },
    )


@router.post("/guardar")
def guardar_viaje(
    solicitud: Request,
    fecha: str = Form(...),
    asientos_disponibles: int = Form(...),
    costo: int = Form(...),
    id_ruta: int = Form(...),
    estado: str = Form(EstadoViaje.PENDIENTE.value),
    id_usuario: int | None = Form(None),
):
    """Ruta HTTP POST para procesar el formulario de creación y almacenar el nuevo viaje."""
    conductor_id = id_usuario if id_usuario is not None else _obtener_id_conductor_actual()
    estado_final = (estado or EstadoViaje.PENDIENTE.value).strip().lower()

    nuevo_viaje = ViajeModel(
        fecha=fecha,
        asientos_disponibles=asientos_disponibles,
        costo=costo,
        estado=estado_final,
        id_ruta=id_ruta,
        id_usuario=conductor_id,
    )

    ViajeController.crear_viaje(nuevo_viaje)

    url_retorno = "/viaje?rol=admin" if _es_admin(solicitud) else "/viaje"
    return RedirectResponse(
        url=url_retorno,
        status_code=status.HTTP_303_SEE_OTHER,
    )


@router.get("/editar/{id_viaje}")
def vista_editar_viaje(solicitud: Request, id_viaje: int):
    """Ruta HTTP GET para cargar los datos de un viaje existente dentro del formulario de edición."""
    viaje_encontrado = ViajeController.obtener_viaje_por_id(id_viaje)
    id_conductor_actual = _obtener_id_conductor_actual()

    rutas_raw = list(db["ruta"].find({}, {"_id": 0}))
    rutas = []
    for r in rutas_raw:
        r["origen"] = r.get("origen", "Origen")
        r["destino"] = r.get("destino", "Destino")
        rutas.append(r)

    es_admin = _es_admin(solicitud)
    plantilla = "admin/viaje_form_admin.html" if es_admin else "viaje/viaje_form.html"

    return motor_plantillas.TemplateResponse(
        solicitud,
        plantilla,
        {
            "viaje": viaje_encontrado,
            "rutas": rutas,
            "id_conductor_actual": id_conductor_actual,
            "es_admin": es_admin,
        },
    )


@router.post("/actualizar")
def actualizar_viaje(
    solicitud: Request,
    id_viaje: int = Form(...),
    fecha: str = Form(None),
    asientos_disponibles: int = Form(None),
    costo: int = Form(None),
    estado: str = Form(None),
    id_ruta: int = Form(None),
    id_usuario: int = Form(None),
):
    """Ruta HTTP POST para procesar la actualización parcial de un viaje existente."""
    viaje_actualizacion = ViajeUpdateModel(
        fecha=fecha,
        asientos_disponibles=asientos_disponibles,
        costo=costo,
        estado=estado.lower() if estado else None,
        id_ruta=id_ruta,
        id_usuario=id_usuario,
    )

    ViajeController.actualizar_viaje(id_viaje, viaje_actualizacion)

    url_retorno = "/viaje?rol=admin" if _es_admin(solicitud) else "/viaje"
    return RedirectResponse(
        url=url_retorno,
        status_code=status.HTTP_303_SEE_OTHER,
    )


@router.get("/eliminar/{id_viaje}")
def eliminar_viaje(solicitud: Request, id_viaje: int):
    """Ruta HTTP GET para procesar la eliminación definitiva de un viaje."""
    ViajeController.eliminar_viaje(id_viaje)

    url_retorno = "/viaje?rol=admin" if _es_admin(solicitud) else "/viaje"
    return RedirectResponse(
        url=url_retorno,
        status_code=status.HTTP_303_SEE_OTHER,
    )