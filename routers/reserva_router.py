from controllers.reserva_controller import ReservaController
from models.reserva_model import ReservaModel, ReservaUpdateModel
from database import db
from fastapi import APIRouter, Form, Request, status
from fastapi.responses import RedirectResponse
from fastapi.templating import Jinja2Templates

# Importamos nuestro servicio de Machine Learning para Velox
from services.ml_service import predecir_confiabilidad_pasajero

router = APIRouter(prefix="/reserva", tags=["Reserva"])
motor_plantillas = Jinja2Templates(directory="templates")


def _es_admin(solicitud: Request) -> bool:
    """Detecta si la petición pertenece a una sesión de Administrador."""
    rol_query = (solicitud.query_params.get("rol") or "").strip().lower()
    rol_header = (solicitud.headers.get("x-velox-rol") or "").strip().lower()
    rol_cookie = (solicitud.cookies.get("velox_usuario_rol") or "").strip().lower()
    return "admin" in rol_query or "admin" in rol_header or "admin" in rol_cookie


def _resolver_ruta_viaje(viaje_doc: dict) -> tuple[str, str]:
    """Obtiene origen y destino de un viaje, buscando en 'ruta' si solo tiene id_ruta."""
    origen = viaje_doc.get("origen") or viaje_doc.get("punto_partida")
    destino = viaje_doc.get("destino") or viaje_doc.get("punto_llegada")

    if (not origen or not destino) and "id_ruta" in viaje_doc:
        ruta_doc = db["ruta"].find_one({"id_ruta": viaje_doc["id_ruta"]}, {"_id": 0})
        if ruta_doc:
            origen = origen or ruta_doc.get("origen") or ruta_doc.get("punto_partida", "Origen")
            destino = destino or ruta_doc.get("destino") or ruta_doc.get("punto_llegada", "Destino")

    return origen or "Origen", destino or "Destino"


@router.get("/")
def listar_reservas(solicitud: Request):
    """Muestra el listado de reservas enriquecidas con nombre, ruta y predicción IA."""
    try:
        lista_reservas_raw = ReservaController.obtener_todas_las_reservas()
    except Exception:
        lista_reservas_raw = list(db["reserva"].find({}, {"_id": 0}))

    reservas_enriquecidas = []

    for reserva in lista_reservas_raw:
        reserva_dict = reserva if isinstance(reserva, dict) else reserva.model_dump()

        # 1. Datos históricos y nombre del pasajero
        usuario = db["usuario"].find_one({"id_usuario": reserva_dict.get("id_usuario")}, {"_id": 0})

        if not reserva_dict.get("nombre_pasajero"):
            if usuario:
                nombre = usuario.get("nombre", "")
                apellido = usuario.get("apellido_paterno", "")
                reserva_dict["nombre_pasajero"] = f"{nombre} {apellido}".strip() or f"Pasajero #{reserva_dict.get('id_usuario')}"
            else:
                reserva_dict["nombre_pasajero"] = f"Pasajero #{reserva_dict.get('id_usuario')}"

        # 2. Datos de origen y destino del viaje
        if not reserva_dict.get("origen") or not reserva_dict.get("destino"):
            viaje_doc = db["viaje"].find_one({"id_viaje": reserva_dict.get("id_viaje")}, {"_id": 0})
            if viaje_doc:
                origen, destino = _resolver_ruta_viaje(viaje_doc)
                reserva_dict["origen"] = reserva_dict.get("origen") or origen
                reserva_dict["destino"] = reserva_dict.get("destino") or destino

        # 3. Evaluación de confiabilidad con Machine Learning (Random Forest)
        historial_asistencia = usuario.get("historial_asistencia", 90.0) if usuario else 90.0
        cancelaciones_previas = usuario.get("cancelaciones", 0) if usuario else 0
        viajes_totales = usuario.get("viajes_totales", 5) if usuario else 5

        reserva_dict["analisis_ia"] = predecir_confiabilidad_pasajero(
            historial_asistencia=historial_asistencia,
            cancelaciones_previas=cancelaciones_previas,
            viajes_totales=viajes_totales,
        )

        reservas_enriquecidas.append(reserva_dict)

    reservas_enriquecidas.sort(key=lambda r: r.get("id_reserva", 0), reverse=True)

    es_admin = _es_admin(solicitud)
    plantilla = "admin/reserva_inicio_admin.html" if es_admin else "reserva/reserva_inicio.html"

    return motor_plantillas.TemplateResponse(
        solicitud,
        plantilla,
        {
            "reservas": reservas_enriquecidas,
            "usuario": {"nombre": "Brandon Rivera"},
            "es_admin": es_admin,
        },
    )


@router.post("/estado/{id_reserva}")
def cambiar_estado_reserva(
    solicitud: Request,
    id_reserva: int,
    nuevo_estado: str = Form(...),
):
    """
    Actualiza el estado de una solicitud ('aceptada' o 'rechazada') desde el panel
    del conductor y sincroniza los asientos disponibles con el panel del pasajero.
    """
    estado_limpio = nuevo_estado.strip().lower()
    reserva_actual = db["reserva"].find_one({"id_reserva": id_reserva})

    if reserva_actual:
        estado_previo = str(reserva_actual.get("estado", "")).lower()

        db["reserva"].update_one(
            {"id_reserva": id_reserva},
            {"$set": {"estado": estado_limpio}},
        )

        if estado_limpio == "aceptada" and estado_previo != "aceptada":
            id_viaje = reserva_actual.get("id_viaje")
            if id_viaje is not None:
                db["viaje"].update_one(
                    {"id_viaje": id_viaje, "asientos_disponibles": {"$gt": 0}},
                    {"$inc": {"asientos_disponibles": -1}},
                )

    url_retorno = "/reserva/?rol=admin" if _es_admin(solicitud) else "/reserva/"
    return RedirectResponse(
        url=url_retorno,
        status_code=status.HTTP_303_SEE_OTHER,
    )


@router.get("/nuevo")
def vista_crear_reserva(solicitud: Request):
    """Carga el formulario para registrar una nueva reserva manualmente."""
    usuarios = list(db["usuario"].find({}, {"_id": 0}))
    viajes_raw = list(db["viaje"].find({}, {"_id": 0}))
    viajes = []
    for v in viajes_raw:
        origen, destino = _resolver_ruta_viaje(v)
        v["origen"] = origen
        v["destino"] = destino
        viajes.append(v)

    es_admin = _es_admin(solicitud)
    plantilla = "admin/reserva_form_admin.html" if es_admin else "reserva/reserva_form.html"

    return motor_plantillas.TemplateResponse(
        solicitud,
        plantilla,
        {
            "reserva": None,
            "usuarios": usuarios,
            "viajes": viajes,
            "es_admin": es_admin,
        },
    )


@router.post("/guardar")
def guardar_reserva(
    solicitud: Request,
    estado: str = Form(...),
    id_usuario: int = Form(...),
    id_viaje: int = Form(...),
):
    """Procesa el formulario de creación y almacena la nueva reserva."""
    nueva_reserva = ReservaModel(
        estado=estado.lower(),
        fecha_solicitud="",
        id_usuario=id_usuario,
        id_viaje=id_viaje,
    )

    ReservaController.crear_reserva(nueva_reserva)

    url_retorno = "/reserva/?rol=admin" if _es_admin(solicitud) else "/reserva/"
    return RedirectResponse(
        url=url_retorno,
        status_code=status.HTTP_303_SEE_OTHER,
    )


@router.get("/editar/{id_reserva}")
def vista_editar_reserva(solicitud: Request, id_reserva: int):
    """Carga los datos de una reserva existente para edición."""
    reserva_encontrada = ReservaController.obtener_reserva_por_id(id_reserva)
    usuarios = list(db["usuario"].find({}, {"_id": 0}))

    viajes_raw = list(db["viaje"].find({}, {"_id": 0}))
    viajes = []
    for v in viajes_raw:
        origen, destino = _resolver_ruta_viaje(v)
        v["origen"] = origen
        v["destino"] = destino
        viajes.append(v)

    es_admin = _es_admin(solicitud)
    plantilla = "admin/reserva_form_admin.html" if es_admin else "reserva/reserva_form.html"

    return motor_plantillas.TemplateResponse(
        solicitud,
        plantilla,
        {
            "reserva": reserva_encontrada,
            "usuarios": usuarios,
            "viajes": viajes,
            "es_admin": es_admin,
        },
    )


@router.post("/actualizar")
def actualizar_reserva(
    solicitud: Request,
    id_reserva: int = Form(...),
    estado: str = Form(None),
    id_usuario: int = Form(None),
    id_viaje: int = Form(None),
):
    """Procesa la actualización parcial de una reserva existente."""
    reserva_actualizacion = ReservaUpdateModel(
        estado=estado.lower() if estado else None,
        id_usuario=id_usuario,
        id_viaje=id_viaje,
    )

    ReservaController.actualizar_reserva(id_reserva, reserva_actualizacion)

    url_retorno = "/reserva/?rol=admin" if _es_admin(solicitud) else "/reserva/"
    return RedirectResponse(
        url=url_retorno,
        status_code=status.HTTP_303_SEE_OTHER,
    )


@router.get("/eliminar/{id_reserva}")
def eliminar_reserva(solicitud: Request, id_reserva: int):
    """Procesa la eliminación definitiva de una reserva."""
    ReservaController.eliminar_reserva(id_reserva)

    url_retorno = "/reserva/?rol=admin" if _es_admin(solicitud) else "/reserva/"
    return RedirectResponse(
        url=url_retorno,
        status_code=status.HTTP_303_SEE_OTHER,
    )