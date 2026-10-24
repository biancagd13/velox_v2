from controllers.reserva_controller import ReservaController
from models.reserva_model import ReservaModel, ReservaUpdateModel
from database import db
from fastapi import APIRouter, Form, Request, status
from fastapi.responses import RedirectResponse
from fastapi.templating import Jinja2Templates

# Inicializamos el enrutador con su prefijo institucional
router = APIRouter(prefix="/reserva", tags=["Reserva"])
motor_plantillas = Jinja2Templates(directory="templates")


@router.get("/")
def listar_reservas(solicitud: Request):
    """Ruta HTTP GET para mostrar la vista principal con el listado de todas las reservas."""
    lista_reservas = ReservaController.obtener_todas_las_reservas()
    return motor_plantillas.TemplateResponse(
        solicitud,
        "reserva/reserva_inicio.html",
        {
            "reservas": lista_reservas,
        },
    )


@router.get("/nuevo")
def vista_crear_reserva(solicitud: Request):
    """Ruta HTTP GET para cargar el formulario vacío destinado a registrar una nueva reserva."""
    usuarios = list(db["usuario"].find({}, {"_id": 0}))
    
    # Consultamos los viajes y resolvemos sus rutas de forma robusta
    viajes_raw = list(db["viaje"].find({}, {"_id": 0}))
    viajes = []
    for v in viajes_raw:
        origen = v.get("origen") or v.get("punto_partida")
        destino = v.get("destino") or v.get("punto_llegada")
        
        if (not origen or not destino) and "id_ruta" in v:
            ruta_doc = db["ruta"].find_one({"id_ruta": v["id_ruta"]}, {"_id": 0})
            if ruta_doc:
                origen = origen or ruta_doc.get("origen") or ruta_doc.get("punto_partida", "Origen")
                destino = destino or ruta_doc.get("destino") or ruta_doc.get("punto_llegada", "Destino")
        
        v["origen"] = origen or "Origen"
        v["destino"] = destino or "Destino"
        viajes.append(v)

    return motor_plantillas.TemplateResponse(
        solicitud, 
        "reserva/reserva_form.html", 
        {
            "reserva": None, 
            "usuarios": usuarios, 
            "viajes": viajes
        }
    )


@router.post("/guardar")
def guardar_reserva(
    estado: str = Form(...),
    id_usuario: int = Form(...),
    id_viaje: int = Form(...)
):
    """Ruta HTTP POST para procesar el formulario de creación y almacenar la nueva reserva."""
    nueva_reserva = ReservaModel(
        estado=estado.lower(),  # Convertimos a minúsculas para cumplir con el Enum de Pydantic
        fecha_solicitud="",  # Se genera automáticamente en el controlador
        id_usuario=id_usuario,
        id_viaje=id_viaje
    )

    ReservaController.crear_reserva(nueva_reserva)

    return RedirectResponse(
        url="/reserva",
        status_code=status.HTTP_303_SEE_OTHER,
    )


@router.get("/editar/{id_reserva}")
def vista_editar_reserva(solicitud: Request, id_reserva: int):
    """Ruta HTTP GET para cargar los datos de una reserva existente dentro del formulario de edición."""
    reserva_encontrada = ReservaController.obtener_reserva_por_id(id_reserva)
    
    usuarios = list(db["usuario"].find({}, {"_id": 0}))
    
    # Enriquecimiento robusto de los viajes también para la vista de edición
    viajes_raw = list(db["viaje"].find({}, {"_id": 0}))
    viajes = []
    for v in viajes_raw:
        origen = v.get("origen") or v.get("punto_partida")
        destino = v.get("destino") or v.get("punto_llegada")
        
        if (not origen or not destino) and "id_ruta" in v:
            ruta_doc = db["ruta"].find_one({"id_ruta": v["id_ruta"]}, {"_id": 0})
            if ruta_doc:
                origen = origen or ruta_doc.get("origen") or ruta_doc.get("punto_partida", "Origen")
                destino = destino or ruta_doc.get("destino") or ruta_doc.get("punto_llegada", "Destino")
        
        v["origen"] = origen or "Origen"
        v["destino"] = destino or "Destino"
        viajes.append(v)

    return motor_plantillas.TemplateResponse(
        solicitud, 
        "reserva/reserva_form.html", 
        {
            "reserva": reserva_encontrada, 
            "usuarios": usuarios, 
            "viajes": viajes
        }
    )


@router.post("/actualizar")
def actualizar_reserva(
    id_reserva: int = Form(...),
    estado: str = Form(None),
    id_usuario: int = Form(None),
    id_viaje: int = Form(None)
):
    """Ruta HTTP POST para procesar la actualización parcial de una reserva existente."""
    reserva_actualizacion = ReservaUpdateModel(
        estado=estado.lower() if estado else None,  # Convertimos a minúsculas de manera segura
        id_usuario=id_usuario,
        id_viaje=id_viaje
    )

    ReservaController.actualizar_reserva(id_reserva, reserva_actualizacion)

    return RedirectResponse(
        url="/reserva",
        status_code=status.HTTP_303_SEE_OTHER,
    )


@router.get("/eliminar/{id_reserva}")
def eliminar_reserva(id_reserva: int):
    """Ruta HTTP GET para procesar la eliminación definitiva de una reserva."""
    ReservaController.eliminar_reserva(id_reserva)

    return RedirectResponse(
        url="/reserva",
        status_code=status.HTTP_303_SEE_OTHER,
    )