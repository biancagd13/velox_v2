from datetime import datetime, timezone

from controllers.reserva_controller import ReservaController
from models.reserva_model import ReservaModel, ReservaUpdateModel
from database import db
from fastapi import APIRouter, Body, Form, Request, status
from fastapi.responses import JSONResponse, RedirectResponse
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


def _to_int_or_none(valor) -> int | None:
    """Convierte a entero si es válido; de lo contrario devuelve None."""
    if valor is None:
        return None
    texto = str(valor).strip()
    if texto.isdigit():
        return int(texto)
    return None


def _extraer_id_usuario(solicitud: Request, payload: dict | None = None) -> int | None:
    raw = (
        (payload.get("id_usuario") if payload else None)
        or solicitud.query_params.get("id_usuario")
        or solicitud.cookies.get("velox_usuario_id")
        or solicitud.headers.get("x-velox-usuario-id")
    )
    return _to_int_or_none(raw)


def _obtener_id_conductor_efectivo(solicitud: Request) -> int | None:
    """Obtiene el ID del conductor activo o un usuario no-admin por defecto si no viene en la petición."""
    id_sesion = _extraer_id_usuario(solicitud)
    if id_sesion is not None and id_sesion > 0:
        return id_sesion

    usuario_no_admin = db["usuario"].find_one(
        {"rol": {"$not": {"$regex": "admin", "$options": "i"}}},
        {"_id": 0, "id_usuario": 1},
    )
    if usuario_no_admin:
        return _to_int_or_none(usuario_no_admin.get("id_usuario"))

    return None


def _obtener_ids_viaje_conductor(id_conductor: int | None) -> set[int]:
    """Obtiene los id_viaje que pertenecen al conductor activo."""
    if not id_conductor or id_conductor <= 0:
        return set()

    rutas_cond = list(
        db["ruta"].find(
            {
                "$or": [
                    {"id_usuario": id_conductor},
                    {"id_conductor": id_conductor},
                    {"usuario_id": id_conductor},
                    {"conductor_id": id_conductor},
                ]
            },
            {"_id": 0, "id_ruta": 1},
        )
    )
    ids_ruta = [
        val for r in rutas_cond if (val := _to_int_or_none(r.get("id_ruta"))) is not None
    ]

    condiciones = [
        {"id_usuario": id_conductor},
        {"id_conductor": id_conductor},
        {"usuario_id": id_conductor},
        {"conductor_id": id_conductor},
    ]
    if ids_ruta:
        condiciones.append({"id_ruta": {"$in": ids_ruta}})

    viajes_cond = list(
        db["viaje"].find({"$or": condiciones}, {"_id": 0, "id_viaje": 1})
    )
    return {
        val for v in viajes_cond if (val := _to_int_or_none(v.get("id_viaje"))) is not None
    }


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

        id_u = _to_int_or_none(r.get("id_usuario"))
        if id_u is not None:
            stats_por_id.setdefault(id_u, []).append(valor)

        nombre_p = str(r.get("nombre_pasajero") or "").strip().lower()
        if nombre_p:
            stats_por_nombre.setdefault(nombre_p, []).append(valor)

    usuarios_enriquecidos = []
    for u in usuarios_raw:
        u_dict = dict(u)
        id_u = _to_int_or_none(u_dict.get("id_usuario"))
        nombre_completo = " ".join(
            p for p in [
                str(u_dict.get("nombre") or "").strip(),
                str(u_dict.get("apellido_paterno") or "").strip(),
                str(u_dict.get("apellido_materno") or "").strip(),
            ] if p
        ).strip().lower()

        valores = []
        if id_u is not None:
            valores.extend(stats_por_id.get(id_u, []))
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


@router.get("/api/historial")
def api_historial_viajes(solicitud: Request):
    """Devuelve el historial de reservas y viajes del usuario enriquecido con ruta, horario y conductor."""
    id_usuario_int = _extraer_id_usuario(solicitud)
    filtro: dict = {}
    if id_usuario_int is not None:
        filtro["$or"] = [
            {"id_usuario": id_usuario_int},
            {"id_pasajero": id_usuario_int},
            {"usuario_id": id_usuario_int},
        ]

    reservas_raw = list(db["reserva"].find(filtro, {"_id": 0}).sort("id_reserva", -1))

    viajes_map = {
        int(v["id_viaje"]): v
        for v in db["viaje"].find({}, {"_id": 0})
        if v.get("id_viaje") is not None and str(v.get("id_viaje")).strip().isdigit()
    }
    rutas_map = {
        int(r["id_ruta"]): r
        for r in db["ruta"].find({}, {"_id": 0})
        if r.get("id_ruta") is not None and str(r.get("id_ruta")).strip().isdigit()
    }
    usuarios_map = {
        int(u["id_usuario"]): u
        for u in db["usuario"].find({}, {"_id": 0})
        if u.get("id_usuario") is not None and str(u.get("id_usuario")).strip().isdigit()
    }

    historial = []
    for res in reservas_raw:
        id_viaje = _to_int_or_none(res.get("id_viaje"))
        viaje_doc = viajes_map.get(id_viaje, {}) if id_viaje is not None else {}

        id_ruta = (
            _to_int_or_none(viaje_doc.get("id_ruta"))
            or _to_int_or_none(res.get("id_ruta"))
        )
        ruta_doc = rutas_map.get(id_ruta, {}) if id_ruta is not None else {}

        origen = (
            res.get("origen")
            or viaje_doc.get("origen")
            or ruta_doc.get("origen")
            or "Origen"
        )
        destino = (
            res.get("destino")
            or viaje_doc.get("destino")
            or ruta_doc.get("destino")
            or "Destino"
        )
        fecha = str(viaje_doc.get("fecha") or res.get("fecha_solicitud") or "")[:10]
        hora_salida = str(viaje_doc.get("hora_salida") or ruta_doc.get("hora_salida") or "--:--").strip()
        costo_raw = viaje_doc.get("costo") if viaje_doc.get("costo") is not None else ruta_doc.get("costo", 0)
        try:
            costo_num = float(costo_raw or 0)
            aportacion = int(costo_num) if costo_num.is_integer() else costo_num
        except (TypeError, ValueError):
            aportacion = 0

        id_cond_raw = (
            viaje_doc.get("id_usuario")
            or viaje_doc.get("id_conductor")
            or viaje_doc.get("usuario_id")
            or ruta_doc.get("id_usuario")
        )
        id_cond = _to_int_or_none(id_cond_raw)
        doc_cond = usuarios_map.get(id_cond, {}) if id_cond is not None else {}
        nombre_conductor = (
            " ".join(
                p for p in [
                    str(doc_cond.get("nombre") or "").strip(),
                    str(doc_cond.get("apellido_paterno") or "").strip(),
                ] if p
            ).strip()
            or str(doc_cond.get("nombre_completo") or "Conductor Velox").strip()
        )
        calificacion_cond = str(doc_cond.get("calificacion") or "5.0")

        estado_reserva = str(res.get("estado") or "pendiente").strip().lower()
        estado_viaje = str(viaje_doc.get("estado") or "pendiente").strip().lower()

        historial.append(
            {
                "id_reserva": res.get("id_reserva"),
                "id_viaje": id_viaje,
                "origen": origen,
                "destino": destino,
                "fecha": fecha,
                "hora_salida": hora_salida,
                "aportacion": aportacion,
                "conductor": nombre_conductor,
                "calificacion": calificacion_cond,
                "estado_reserva": estado_reserva,
                "estado_viaje": estado_viaje,
                "fecha_solicitud": res.get("fecha_solicitud") or "",
            }
        )

    return JSONResponse(content={"ok": True, "historial": historial})


@router.get("/api/mis-reservas")
def api_mis_reservas(solicitud: Request):
    """Devuelve las reservas activas/recientes del usuario en formato JSON."""
    id_usuario_int = _extraer_id_usuario(solicitud)
    filtro: dict = {
        "estado": {"$nin": ["cancelada", "cancelado", "rechazada", "rechazado"]}
    }
    if id_usuario_int is not None:
        filtro["$or"] = [
            {"id_usuario": id_usuario_int},
            {"id_pasajero": id_usuario_int},
            {"usuario_id": id_usuario_int},
        ]

    reservas = list(db["reserva"].find(filtro, {"_id": 0}).sort("id_reserva", -1))
    return JSONResponse(content={"ok": True, "reservas": reservas})


@router.post("/api/cancelar/{id_reserva}")
def cancelar_reserva_api(id_reserva: int):
    """Cancela una solicitud o reserva del pasajero y libera el asiento si corresponde."""
    reserva_doc = db["reserva"].find_one({"id_reserva": id_reserva})
    if not reserva_doc:
        return JSONResponse(
            status_code=404,
            content={"ok": False, "mensaje": "La solicitud o reserva no existe."},
        )

    estado_previo = str(reserva_doc.get("estado") or "").strip().lower()
    if estado_previo in {"cancelada", "cancelado"}:
        return JSONResponse(
            status_code=400,
            content={"ok": False, "mensaje": "Esta solicitud ya había sido cancelada."},
        )
    if estado_previo in {"finalizada", "finalizado", "completada", "completado"}:
        return JSONResponse(
            status_code=400,
            content={"ok": False, "mensaje": "No puedes cancelar una reserva ya finalizada."},
        )

    db["reserva"].update_one(
        {"id_reserva": id_reserva},
        {
            "$set": {
                "estado": "cancelada",
                "fecha_cancelacion": datetime.now(timezone.utc).isoformat(),
            }
        },
    )

    id_viaje = reserva_doc.get("id_viaje")
    if id_viaje is not None:
        try:
            id_viaje_int = int(id_viaje)
            viaje_doc = db["viaje"].find_one({"id_viaje": id_viaje_int}, {"_id": 0})
            if viaje_doc and viaje_doc.get("asientos_disponibles") is not None:
                actual = int(viaje_doc.get("asientos_disponibles") or 0)
                capacidad = viaje_doc.get("capacidad_lugares") or viaje_doc.get("lugares")
                if capacidad is None or actual < int(capacidad):
                    db["viaje"].update_one(
                        {"id_viaje": id_viaje_int},
                        {"$inc": {"asientos_disponibles": 1}},
                    )
        except (TypeError, ValueError):
            pass

    return JSONResponse(
        content={
            "ok": True,
            "id_reserva": id_reserva,
            "mensaje": "Tu solicitud fue cancelada correctamente.",
        }
    )


@router.post("/api")
def crear_reserva_api(solicitud: Request, payload: dict = Body(...)):
    """Registra una solicitud de reserva desde la vista del pasajero y actualiza cupo."""
    id_viaje = payload.get("id_viaje")
    try:
        id_viaje_int = int(id_viaje)
    except (TypeError, ValueError):
        return JSONResponse(
            status_code=400,
            content={"ok": False, "mensaje": "ID de viaje no válido."},
        )

    id_usuario_int = _extraer_id_usuario(solicitud, payload)

    viaje_doc = db["viaje"].find_one({"id_viaje": id_viaje_int})
    if not viaje_doc:
        return JSONResponse(
            status_code=404,
            content={"ok": False, "mensaje": "El viaje seleccionado no existe."},
        )

    if id_usuario_int is not None:
        reserva_existente = db["reserva"].find_one(
            {
                "id_viaje": id_viaje_int,
                "estado": {
                    "$nin": ["cancelada", "cancelado", "rechazada", "rechazado"]
                },
                "$or": [
                    {"id_usuario": id_usuario_int},
                    {"id_pasajero": id_usuario_int},
                    {"usuario_id": id_usuario_int},
                ],
            }
        )
        if reserva_existente:
            return JSONResponse(
                status_code=400,
                content={
                    "ok": False,
                    "mensaje": "Ya enviaste una solicitud o ya tienes un lugar reservado en este viaje.",
                },
            )

    asientos = int(viaje_doc.get("asientos_disponibles") or 0)
    if asientos <= 0:
        return JSONResponse(
            status_code=400,
            content={"ok": False, "mensaje": "Ya no hay lugares disponibles en este viaje."},
        )

    origen, destino = _resolver_ruta_viaje(viaje_doc)
    ultima_reserva = db["reserva"].find_one({}, sort=[("id_reserva", -1)])
    siguiente_id = (
        int(ultima_reserva["id_reserva"]) + 1
        if ultima_reserva and ultima_reserva.get("id_reserva") is not None
        else 1
    )

    nombre_pasajero = ""
    if id_usuario_int is not None:
        doc_pas = db["usuario"].find_one({"id_usuario": id_usuario_int}, {"_id": 0})
        if doc_pas:
            nombre_pasajero = " ".join(
                p for p in [
                    str(doc_pas.get("nombre") or "").strip(),
                    str(doc_pas.get("apellido_paterno") or "").strip(),
                    str(doc_pas.get("apellido_materno") or "").strip(),
                ] if p
            ).strip()

    nueva_reserva = {
        "id_reserva": siguiente_id,
        "id_viaje": id_viaje_int,
        "id_ruta": viaje_doc.get("id_ruta"),
        "id_usuario": id_usuario_int,
        "id_pasajero": id_usuario_int,
        "nombre_pasajero": nombre_pasajero,
        "origen": origen,
        "destino": destino,
        "estado": "pendiente",
        "fecha_solicitud": datetime.now(timezone.utc).isoformat(),
    }

    db["reserva"].insert_one(nueva_reserva)
    db["viaje"].update_one(
        {"id_viaje": id_viaje_int},
        {"$inc": {"asientos_disponibles": -1}},
    )

    return JSONResponse(
        status_code=201,
        content={
            "ok": True,
            "id_reserva": siguiente_id,
            "mensaje": "Tu solicitud de lugar quedó registrada en la base de datos.",
        },
    )


@router.get("/")
def listar_reservas(solicitud: Request):
    """Muestra el listado de reservas enriquecidas con nombre, ruta y predicción IA según rol."""
    es_admin = _es_admin(solicitud)
    id_conductor_activo = None if es_admin else _obtener_id_conductor_efectivo(solicitud)
    ids_viaje_conductor = set() if es_admin else _obtener_ids_viaje_conductor(id_conductor_activo)

    try:
        lista_reservas_raw = ReservaController.obtener_todas_las_reservas()
    except Exception:
        lista_reservas_raw = list(db["reserva"].find({}, {"_id": 0}))

    reservas_enriquecidas = []

    for reserva in lista_reservas_raw:
        reserva_dict = reserva if isinstance(reserva, dict) else reserva.model_dump()
        id_viaje_res = _to_int_or_none(reserva_dict.get("id_viaje"))

        if not es_admin:
            if id_viaje_res is None or id_viaje_res not in ids_viaje_conductor:
                continue

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
            viaje_doc = db["viaje"].find_one({"id_viaje": id_viaje_res}, {"_id": 0})
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

    plantilla = "admin/reserva_inicio_admin.html" if es_admin else "reserva/reserva_inicio.html"

    return motor_plantillas.TemplateResponse(
        solicitud,
        plantilla,
        {
            "reservas": reservas_enriquecidas,
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