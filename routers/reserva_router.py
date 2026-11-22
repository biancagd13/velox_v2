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

ESTADOS_ACTIVOS_ASIENTO = {"aceptada", "confirmada"}


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


def _enriquecer_usuarios_con_calificaciones(usuarios_raw: list[dict]) -> list[dict]:
    """Agrega promedio_calificacion y total_calificaciones desde db['calificacion_pasajero'] o usuario."""
    resenas_raw = list(db["calificacion_pasajero"].find({}, {"_id": 0}))
    stats_por_id: dict[int, list[float]] = {}
    stats_por_nombre: dict[str, list[float]] = {}

    for r in resenas_raw:
        estrellas = r.get("estrellas")
        if estrellas is None:
            continue
        try:
            valor = float(estrellas)
        except (TypeError, ValueError):
            continue

        id_u = r.get("id_usuario")
        if id_u is not None and str(id_u).strip().isdigit():
            stats_por_id.setdefault(int(id_u), []).append(valor)

        nombre_p = str(r.get("nombre_pasajero") or "").strip().lower()
        if nombre_p:
            stats_por_nombre.setdefault(nombre_p, []).append(valor)

    usuarios_enriquecidos = []
    for u in usuarios_raw:
        u_dict = dict(u)
        id_u = u_dict.get("id_usuario")
        nombre_completo = " ".join(
            p for p in [
                str(u_dict.get("nombre") or "").strip(),
                str(u_dict.get("apellido_paterno") or "").strip(),
                str(u_dict.get("apellido_materno") or "").strip(),
            ] if p
        ).strip().lower()

        valores = []
        if id_u is not None and str(id_u).strip().isdigit():
            valores.extend(stats_por_id.get(int(id_u), []))
        if not valores and nombre_completo:
            valores.extend(stats_por_nombre.get(nombre_completo, []))

        if valores:
            u_dict["promedio_calificacion"] = round(sum(valores) / len(valores), 1)
            u_dict["total_calificaciones"] = len(valores)
        elif u_dict.get("calificacion") is not None:
            u_dict["promedio_calificacion"] = u_dict.get("calificacion")
            u_dict["total_calificaciones"] = u_dict.get("total_calificaciones", 0)
        else:
            u_dict["promedio_calificacion"] = None
            u_dict["total_calificaciones"] = 0

        usuarios_enriquecidos.append(u_dict)

    return usuarios_enriquecidos


def _preparar_viajes_con_ruta() -> list[dict]:
    """Obtiene viajes de db['viaje'] con origen, destino y asientos_disponibles consistentes."""
    viajes_raw = list(db["viaje"].find({}, {"_id": 0}))
    viajes = []
    for v in viajes_raw:
        v_dict = dict(v)
        origen, destino = _resolver_ruta_viaje(v_dict)
        v_dict["origen"] = origen
        v_dict["destino"] = destino

        asientos = v_dict.get("asientos_disponibles")
        if asientos is None:
            asientos = v_dict.get("capacidad_lugares") if v_dict.get("capacidad_lugares") is not None else v_dict.get("lugares")
        v_dict["asientos_disponibles"] = asientos
        viajes.append(v_dict)
    return viajes


def _ajustar_asientos_por_cambio_estado(id_viaje: int | None, estado_previo: str, estado_nuevo: str):
    """Ajusta asientos_disponibles en db['viaje'] al confirmar/aceptar o cancelar/rechazar una reserva."""
    if id_viaje is None:
        return

    prev_activo = estado_previo.strip().lower() in ESTADOS_ACTIVOS_ASIENTO
    nuevo_activo = estado_nuevo.strip().lower() in ESTADOS_ACTIVOS_ASIENTO

    if not prev_activo and nuevo_activo:
        db["viaje"].update_one(
            {"id_viaje": id_viaje, "asientos_disponibles": {"$gt": 0}},
            {"$inc": {"asientos_disponibles": -1}},
        )
    elif prev_activo and not nuevo_activo:
        viaje_doc = db["viaje"].find_one({"id_viaje": id_viaje}, {"_id": 0})
        if viaje_doc and viaje_doc.get("asientos_disponibles") is not None:
            actual = int(viaje_doc.get("asientos_disponibles") or 0)
            capacidad = viaje_doc.get("capacidad_lugares") or viaje_doc.get("lugares")
            if capacidad is None or actual < int(capacidad):
                db["viaje"].update_one(
                    {"id_viaje": id_viaje},
                    {"$inc": {"asientos_disponibles": 1}},
                )


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

        _ajustar_asientos_por_cambio_estado(
            id_viaje=reserva_actual.get("id_viaje"),
            estado_previo=estado_previo,
            estado_nuevo=estado_limpio,
        )

    url_retorno = "/reserva/?rol=admin" if _es_admin(solicitud) else "/reserva/"
    return RedirectResponse(
        url=url_retorno,
        status_code=status.HTTP_303_SEE_OTHER,
    )


@router.get("/nuevo")
def vista_crear_reserva(solicitud: Request):
    """Carga el formulario para registrar una nueva reserva con datos relacionados de BD."""
    usuarios = _enriquecer_usuarios_con_calificaciones(list(db["usuario"].find({}, {"_id": 0})))
    viajes = _preparar_viajes_con_ruta()

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
    """Procesa el formulario de creación, almacena la nueva reserva y sincroniza asientos."""
    estado_limpio = estado.strip().lower()
    nueva_reserva = ReservaModel(
        estado=estado_limpio,
        fecha_solicitud="",
        id_usuario=id_usuario,
        id_viaje=id_viaje,
    )

    ReservaController.crear_reserva(nueva_reserva)
    _ajustar_asientos_por_cambio_estado(
        id_viaje=id_viaje,
        estado_previo="",
        estado_nuevo=estado_limpio,
    )

    url_retorno = "/reserva/?rol=admin" if _es_admin(solicitud) else "/reserva/"
    return RedirectResponse(
        url=url_retorno,
        status_code=status.HTTP_303_SEE_OTHER,
    )


@router.get("/editar/{id_reserva}")
def vista_editar_reserva(solicitud: Request, id_reserva: int):
    """Carga los datos de una reserva existente para edición con datos relacionados de BD."""
    reserva_encontrada = ReservaController.obtener_reserva_por_id(id_reserva)
    usuarios = _enriquecer_usuarios_con_calificaciones(list(db["usuario"].find({}, {"_id": 0})))
    viajes = _preparar_viajes_con_ruta()

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
    """Procesa la actualización parcial de una reserva existente y sincroniza asientos."""
    reserva_previa = db["reserva"].find_one({"id_reserva": id_reserva}, {"_id": 0}) or {}
    estado_previo = str(reserva_previa.get("estado") or "").lower()
    viaje_previo = reserva_previa.get("id_viaje")

    estado_nuevo = estado.strip().lower() if estado else estado_previo
    viaje_nuevo = id_viaje if id_viaje is not None else viaje_previo

    reserva_actualizacion = ReservaUpdateModel(
        estado=estado.lower() if estado else None,
        id_usuario=id_usuario,
        id_viaje=id_viaje,
    )

    ReservaController.actualizar_reserva(id_reserva, reserva_actualizacion)

    if viaje_previo == viaje_nuevo:
        _ajustar_asientos_por_cambio_estado(viaje_nuevo, estado_previo, estado_nuevo)
    else:
        _ajustar_asientos_por_cambio_estado(viaje_previo, estado_previo, "cancelada")
        _ajustar_asientos_por_cambio_estado(viaje_nuevo, "", estado_nuevo)

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