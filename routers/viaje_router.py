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


def _obtener_id_conductor_actual() -> int:
    """Función auxiliar que obtiene el id_usuario del conductor autenticado desde MongoDB."""
    # Buscamos primero la cuenta institucional del conductor activo en la colección 'usuario'
    conductor = db["usuario"].find_one(
        {"correo_institucional": "al22288@utvt.edu.mx"},
        {"_id": 0, "id_usuario": 1},
    )
    if conductor and conductor.get("id_usuario") is not None:
        return int(conductor["id_usuario"])

    # Si no se encuentra por correo, devolvemos el identificador base del conductor (1)
    return 1


@router.get("/")
def listar_viajes(solicitud: Request):
    """Ruta HTTP GET para mostrar la vista principal con el listado de todos los viajes."""
    # Consultamos todos los viajes registrados mediante la clase ViajeController
    lista_viajes = ViajeController.obtener_todos_los_viajes()

    # Renderizamos la plantilla principal enviando el listado obtenido
    return motor_plantillas.TemplateResponse(
        solicitud,
        "viaje/viaje_inicio.html",
        {
            "viajes": lista_viajes,
        },
    )


@router.get("/nuevo")
def vista_crear_viaje(solicitud: Request):
    """Ruta HTTP GET para cargar el formulario vacío destinado a registrar un nuevo viaje."""
    # Obtenemos el ID del conductor autenticado para enviarlo de forma oculta en el formulario
    id_conductor_actual = _obtener_id_conductor_actual()

    # Consultamos las rutas disponibles y normalizamos origen y destino para el selector
    rutas_raw = list(db["ruta"].find({}, {"_id": 0}))
    rutas = []
    for r in rutas_raw:
        r["origen"] = r.get("origen", "Origen")
        r["destino"] = r.get("destino", "Destino")
        rutas.append(r)

    # Renderizamos el formulario sin exponer selectores innecesarios de estado ni conductor
    return motor_plantillas.TemplateResponse(
        solicitud,
        "viaje/viaje_form.html",
        {
            "viaje": None,
            "rutas": rutas,
            "id_conductor_actual": id_conductor_actual,
        },
    )


@router.post("/guardar")
def guardar_viaje(
    fecha: str = Form(...),
    asientos_disponibles: int = Form(...),
    costo: int = Form(...),
    id_ruta: int = Form(...),
    estado: str = Form(EstadoViaje.PENDIENTE.value),
    id_usuario: int | None = Form(None),
):
    """Ruta HTTP POST para procesar el formulario de creación y almacenar el nuevo viaje."""
    # Asignamos automáticamente el conductor autenticado si el formulario no envió un ID explícito
    conductor_id = id_usuario if id_usuario is not None else _obtener_id_conductor_actual()

    # Forzamos el estado inicial 'pendiente' en minúsculas conforme a EstadoViaje
    estado_final = (estado or EstadoViaje.PENDIENTE.value).strip().lower()

    # Empaquetamos y validamos los datos en el modelo Pydantic ViajeModel
    nuevo_viaje = ViajeModel(
        fecha=fecha,
        asientos_disponibles=asientos_disponibles,
        costo=costo,
        estado=estado_final,
        id_ruta=id_ruta,
        id_usuario=conductor_id,
    )

    # Delegamos la inserción en MongoDB a la clase controladora
    ViajeController.crear_viaje(nuevo_viaje)

    # Redirigimos al listado principal de viajes usando código HTTP 303 (See Other)
    return RedirectResponse(
        url="/viaje",
        status_code=status.HTTP_303_SEE_OTHER,
    )


@router.get("/editar/{id_viaje}")
def vista_editar_viaje(solicitud: Request, id_viaje: int):
    """Ruta HTTP GET para cargar los datos de un viaje existente dentro del formulario de edición."""
    # Buscamos el registro del viaje solicitado utilizando ViajeController
    viaje_encontrado = ViajeController.obtener_viaje_por_id(id_viaje)

    # Obtenemos el ID del conductor autenticado como respaldo para el campo oculto
    id_conductor_actual = _obtener_id_conductor_actual()

    # Consultamos el catálogo de rutas para llenar el selector en modo edición
    rutas_raw = list(db["ruta"].find({}, {"_id": 0}))
    rutas = []
    for r in rutas_raw:
        r["origen"] = r.get("origen", "Origen")
        r["destino"] = r.get("destino", "Destino")
        rutas.append(r)

    # Renderizamos el formulario precargado con los datos del viaje
    return motor_plantillas.TemplateResponse(
        solicitud,
        "viaje/viaje_form.html",
        {
            "viaje": viaje_encontrado,
            "rutas": rutas,
            "id_conductor_actual": id_conductor_actual,
        },
    )


@router.post("/actualizar")
def actualizar_viaje(
    id_viaje: int = Form(...),
    fecha: str = Form(None),
    asientos_disponibles: int = Form(None),
    costo: int = Form(None),
    estado: str = Form(None),
    id_ruta: int = Form(None),
    id_usuario: int = Form(None),
):
    """Ruta HTTP POST para procesar la actualización parcial de un viaje existente."""
    # Empaquetamos los campos recibidos dentro del modelo de actualización parcial
    viaje_actualizacion = ViajeUpdateModel(
        fecha=fecha,
        asientos_disponibles=asientos_disponibles,
        costo=costo,
        estado=estado.lower() if estado else None,
        id_ruta=id_ruta,
        id_usuario=id_usuario,
    )

    # Delegamos la actualización en base de datos a la clase controladora
    ViajeController.actualizar_viaje(id_viaje, viaje_actualizacion)

    # Redirigimos de vuelta al listado de viajes
    return RedirectResponse(
        url="/viaje",
        status_code=status.HTTP_303_SEE_OTHER,
    )


@router.get("/eliminar/{id_viaje}")
def eliminar_viaje(id_viaje: int):
    """Ruta HTTP GET para procesar la eliminación definitiva de un viaje."""
    # Solicitamos al controlador eliminar el viaje indicado por su identificador
    ViajeController.eliminar_viaje(id_viaje)

    # Redirigimos al listado principal una vez completada la baja
    return RedirectResponse(
        url="/viaje",
        status_code=status.HTTP_303_SEE_OTHER,
    )