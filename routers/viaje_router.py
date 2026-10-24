from controllers.viaje_controller import ViajeController
from models.viaje_model import ViajeModel, ViajeUpdateModel
from database import db
from fastapi import APIRouter, Form, Request, status
from fastapi.responses import RedirectResponse
from fastapi.templating import Jinja2Templates

# Inicializamos el enrutador con el prefijo y etiqueta correspondientes en español
router = APIRouter(prefix="/viaje", tags=["Viaje"])
motor_plantillas = Jinja2Templates(directory="templates")


@router.get("/")
def listar_viajes(solicitud: Request):
    """Ruta HTTP GET para mostrar la vista principal con el listado de todos los viajes."""
    lista_viajes = ViajeController.obtener_todos_los_viajes()
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
    usuarios = list(db["usuario"].find({}, {"_id": 0}))
    
    # Consultamos las rutas y resolvemos origen y destino para el selector
    rutas_raw = list(db["ruta"].find({}, {"_id": 0}))
    rutas = []
    for r in rutas_raw:
        r["origen"] = r.get("origen", "Origen")
        r["destino"] = r.get("destino", "Destino")
        rutas.append(r)

    return motor_plantillas.TemplateResponse(
        solicitud, 
        "viaje/viaje_form.html", 
        {
            "viaje": None,
            "usuarios": usuarios,
            "rutas": rutas
        }
    )


@router.post("/guardar")
def guardar_viaje(
    fecha: str = Form(...),
    asientos_disponibles: int = Form(...),
    costo: int = Form(...),
    estado: str = Form("pendiente"),
    id_ruta: int = Form(...),
    id_usuario: int = Form(...)
):
    """Ruta HTTP POST para procesar el formulario de creación y almacenar el nuevo viaje."""
    nuevo_viaje = ViajeModel(
        fecha=fecha,
        asientos_disponibles=asientos_disponibles,
        costo=costo,
        estado=estado.lower(),
        id_ruta=id_ruta,
        id_usuario=id_usuario
    )

    ViajeController.crear_viaje(nuevo_viaje)

    return RedirectResponse(
        url="/viaje",
        status_code=status.HTTP_303_SEE_OTHER,
    )


@router.get("/editar/{id_viaje}")
def vista_editar_viaje(solicitud: Request, id_viaje: int):
    """Ruta HTTP GET para cargar los datos de un viaje existente dentro del formulario de edición."""
    viaje_encontrado = ViajeController.obtener_viaje_por_id(id_viaje)
    usuarios = list(db["usuario"].find({}, {"_id": 0}))
    
    rutas_raw = list(db["ruta"].find({}, {"_id": 0}))
    rutas = []
    for r in rutas_raw:
        r["origen"] = r.get("origen", "Origen")
        r["destino"] = r.get("destino", "Destino")
        rutas.append(r)

    return motor_plantillas.TemplateResponse(
        solicitud, 
        "viaje/viaje_form.html", 
        {
            "viaje": viaje_encontrado,
            "usuarios": usuarios,
            "rutas": rutas
        }
    )


@router.post("/actualizar")
def actualizar_viaje(
    id_viaje: int = Form(...),
    fecha: str = Form(None),
    asientos_disponibles: int = Form(None),
    costo: int = Form(None),
    estado: str = Form(None),
    id_ruta: int = Form(None),
    id_usuario: int = Form(None)
):
    """Ruta HTTP POST para procesar la actualización parcial de un viaje existente."""
    viaje_actualizacion = ViajeUpdateModel(
        fecha=fecha,
        asientos_disponibles=asientos_disponibles,
        costo=costo,
        estado=estado.lower() if estado else None,
        id_ruta=id_ruta,
        id_usuario=id_usuario
    )

    ViajeController.actualizar_viaje(id_viaje, viaje_actualizacion)

    return RedirectResponse(
        url="/viaje",
        status_code=status.HTTP_303_SEE_OTHER,
    )


@router.get("/eliminar/{id_viaje}")
def eliminar_viaje(id_viaje: int):
    """Ruta HTTP GET para procesar la eliminación definitiva de un viaje."""
    ViajeController.eliminar_viaje(id_viaje)

    return RedirectResponse(
        url="/viaje",
        status_code=status.HTTP_303_SEE_OTHER,
    )