# Importamos la clase controladora completa para gestionar la lógica de negocio de las calificaciones
from controllers.calificacion_controller import CalificacionController

# Importamos los modelos de Pydantic necesarios para validar los datos de la calificación
from models.calificacion_model import CalificacionModel, CalificacionUpdateModel

# Importamos la base de datos para poder consultar los usuarios, viajes y rutas registrados
from database import db

# Importamos las herramientas de FastAPI: APIRouter para rutas, Form para datos de formularios y Request para solicitudes HTTP
from fastapi import APIRouter, Form, Request, status

# Importamos RedirectResponse para redirigir entre endpoints y Jinja2Templates para renderizar plantillas HTML
from fastapi.responses import RedirectResponse
from fastapi.templating import Jinja2Templates

# Inicializamos el enrutador con el prefijo y etiqueta correspondientes en español
router = APIRouter(prefix="/calificacion", tags=["Calificación"])

# Configuramos el directorio base donde se encuentran las plantillas HTML (carpeta 'templates')
motor_plantillas = Jinja2Templates(directory="templates")


def _es_admin(solicitud: Request) -> bool:
    """Detecta si la petición pertenece a una sesión de Administrador."""
    rol_query = (solicitud.query_params.get("rol") or "").strip().lower()
    rol_header = (solicitud.headers.get("x-velox-rol") or "").strip().lower()
    rol_cookie = (solicitud.cookies.get("velox_usuario_rol") or "").strip().lower()
    return "admin" in rol_query or "admin" in rol_header or "admin" in rol_cookie


def _obtener_id_usuario_sesion(solicitud: Request) -> int | None:
    """Obtiene el ID numérico del usuario activo si viene en query, header o cookie."""
    raw = (
        solicitud.query_params.get("id_usuario")
        or solicitud.headers.get("x-velox-usuario-id")
        or solicitud.cookies.get("velox_usuario_id")
        or ""
    ).strip()
    if raw.isdigit():
        return int(raw)
    return None


def _to_int_or_none(valor) -> int | None:
    """Convierte a entero si es válido; de lo contrario devuelve None."""
    if valor is None:
        return None
    texto = str(valor).strip()
    if texto.isdigit():
        return int(texto)
    return None


def _obtener_id_conductor_actual(solicitud: Request) -> int | None:
    """Obtiene el id_usuario del conductor activo en sesión o desde MongoDB."""
    id_sesion = _obtener_id_usuario_sesion(solicitud)
    if id_sesion and id_sesion > 0:
        return id_sesion

    conductor = db["usuario"].find_one(
        {"$or": [{"id_usuario": 5}, {"correo_institucional": "al222050411@utvtol.edu.mx"}]},
        {"_id": 0, "id_usuario": 1},
    )
    if conductor and conductor.get("id_usuario") is not None:
        id_cond = _to_int_or_none(conductor.get("id_usuario"))
        if id_cond:
            return id_cond

    return None


def _obtener_ids_viaje_y_ruta_conductor(id_conductor: int | None) -> tuple[set[int], set[int]]:
    """Obtiene los id_viaje e id_ruta que pertenecen al conductor activo en MongoDB."""
    if not id_conductor or id_conductor <= 0:
        return set(), set()

    filtro_dueno = {
        "$or": [
            {"id_usuario": id_conductor},
            {"id_conductor": id_conductor},
            {"usuario_id": id_conductor},
            {"conductor_id": id_conductor},
        ]
    }

    rutas_cond = list(db["ruta"].find(filtro_dueno, {"_id": 0, "id_ruta": 1}))
    ids_ruta = set()
    for r in rutas_cond:
        id_r = _to_int_or_none(r.get("id_ruta"))
        if id_r is not None:
            ids_ruta.add(id_r)

    condiciones_viaje = [
        {"id_usuario": id_conductor},
        {"id_conductor": id_conductor},
        {"usuario_id": id_conductor},
        {"conductor_id": id_conductor},
    ]
    if ids_ruta:
        condiciones_viaje.append({"id_ruta": {"$in": list(ids_ruta)}})

    viajes_cond = list(
        db["viaje"].find({"$or": condiciones_viaje}, {"_id": 0, "id_viaje": 1, "id_ruta": 1})
    )
    ids_viaje = set()
    for v in viajes_cond:
        id_v = _to_int_or_none(v.get("id_viaje"))
        if id_v is not None:
            ids_viaje.add(id_v)
        id_r_viaje = _to_int_or_none(v.get("id_ruta"))
        if id_r_viaje is not None:
            ids_ruta.add(id_r_viaje)

    return ids_viaje, ids_ruta


def _pertenece_a_conductor(
    doc: dict,
    id_conductor: int | None,
    ids_viaje: set[int],
    ids_ruta: set[int],
) -> bool:
    """Determina si una reseña está relacionada con el conductor activo."""
    if not id_conductor or id_conductor <= 0:
        return False

    for campo in ("id_conductor", "id_conductor_viaje", "id_usuario_conductor", "conductor_id"):
        val_id = _to_int_or_none(doc.get(campo))
        if val_id is not None and val_id == id_conductor:
            return True

    id_v = _to_int_or_none(doc.get("id_viaje"))
    if id_v is not None and id_v in ids_viaje:
        return True

    id_r = _to_int_or_none(doc.get("id_ruta"))
    if id_r is not None and id_r in ids_ruta:
        return True

    return False


def _resolver_nombre_pasajero(doc: dict, usuarios_map: dict[int, dict]) -> str:
    """Obtiene el nombre real del pasajero desde la reseña o desde db['usuario']."""
    nombre_guardado = str(doc.get("nombre_pasajero") or doc.get("pasajero") or "").strip()
    if nombre_guardado and not nombre_guardado.lower().startswith("usuario"):
        return nombre_guardado

    id_u = _to_int_or_none(doc.get("id_usuario") or doc.get("id_pasajero"))
    if id_u is not None and id_u in usuarios_map:
        usuario = usuarios_map[id_u]
        completo = " ".join(
            p
            for p in [
                str(usuario.get("nombre") or "").strip(),
                str(usuario.get("apellido_paterno") or "").strip(),
                str(usuario.get("apellido_materno") or "").strip(),
            ]
            if p
        ).strip()
        if completo:
            return completo
        if usuario.get("correo_institucional"):
            return str(usuario.get("correo_institucional")).strip()

    if id_u is not None:
        return f"Usuario #{id_u}"
    return "Pasajero"


def _extraer_estrellas(doc: dict) -> int:
    """Normaliza la puntuación recibida en un entero válido de 1 a 5 estrellas."""
    valor = doc.get("estrellas")
    if valor is None:
        valor = doc.get("puntuacion")
    if valor is None:
        valor = doc.get("calificacion")

    try:
        return max(1, min(5, int(round(float(valor)))))
    except (TypeError, ValueError):
        return 5


def _ordenar_por_id(doc: dict) -> int:
    """Ordena de forma segura las reseñas de la más reciente a la más antigua."""
    try:
        return int(doc.get("id_calificacion", 0))
    except (TypeError, ValueError):
        return 0


def _enriquecer_viajes_con_ruta() -> list[dict]:
    """Obtiene todos los viajes de db['viaje'] con origen y destino resueltos."""
    viajes_raw = list(db["viaje"].find({}, {"_id": 0}))
    viajes = []
    for v in viajes_raw:
        origen = v.get("origen") or v.get("punto_partida")
        destino = v.get("destino") or v.get("punto_llegada")
        if (not origen or not destino) and v.get("id_ruta") is not None:
            ruta_doc = db["ruta"].find_one({"id_ruta": v["id_ruta"]}, {"_id": 0})
            if ruta_doc:
                origen = origen or ruta_doc.get("origen") or ruta_doc.get("punto_partida", "Origen")
                destino = destino or ruta_doc.get("destino") or ruta_doc.get("punto_llegada", "Destino")
        v["origen"] = origen or "Origen"
        v["destino"] = destino or "Destino"
        viajes.append(v)
    return viajes


@router.get("/")
def listar_calificaciones(solicitud: Request):
    """Ruta HTTP GET para mostrar la vista principal de calificaciones."""
    lista_raw = CalificacionController.obtener_todas_las_calificaciones()
    es_admin = _es_admin(solicitud)

    if es_admin:
        calificaciones_admin = [
            item.model_dump()
            if hasattr(item, "model_dump")
            else (item.dict() if hasattr(item, "dict") else dict(item))
            for item in lista_raw
        ]
        calificaciones_admin.sort(key=_ordenar_por_id, reverse=True)
        return motor_plantillas.TemplateResponse(
            solicitud,
            "admin/calificacion_inicio_admin.html",
            {
                "calificaciones": calificaciones_admin,
                "total_resenas": len(calificaciones_admin),
                "es_admin": True,
            },
        )

    id_conductor = _obtener_id_conductor_actual(solicitud)
    ids_viaje, ids_ruta = _obtener_ids_viaje_y_ruta_conductor(id_conductor)
    usuarios_map = {
        int(u["id_usuario"]): u
        for u in db["usuario"].find({}, {"_id": 0})
        if u.get("id_usuario") is not None and str(u.get("id_usuario")).strip().isdigit()
    }

    lista_calificaciones = []
    suma_estrellas = 0

    for idx, item in enumerate(lista_raw):
        doc = (
            item.model_dump()
            if hasattr(item, "model_dump")
            else (item.dict() if hasattr(item, "dict") else dict(item))
        )

        if doc.get("tipo") == "conductor_a_pasajero":
            continue

        if not _pertenece_a_conductor(doc, id_conductor, ids_viaje, ids_ruta):
            continue

        estrellas = _extraer_estrellas(doc)
        suma_estrellas += estrellas
        nombre_pasajero = _resolver_nombre_pasajero(doc, usuarios_map)
        id_cal = doc.get("id_calificacion", idx + 1)

        if estrellas >= 4:
            etiqueta, clase_badge, icono = (
                "Excelente",
                "bg-success-subtle text-success",
                "bi-emoji-smile-fill",
            )
        elif estrellas == 3:
            etiqueta, clase_badge, icono = (
                "Regular",
                "bg-warning-subtle text-dark",
                "bi-emoji-neutral-fill",
            )
        else:
            etiqueta, clase_badge, icono = (
                "A mejorar",
                "bg-danger-subtle text-danger",
                "bi-emoji-frown-fill",
            )

        lista_calificaciones.append(
            {
                "id_calificacion": id_cal,
                "nombre_pasajero": nombre_pasajero,
                "inicial": nombre_pasajero[0].upper() if nombre_pasajero else "P",
                "estrellas": estrellas,
                "comentario": doc.get("comentario") or "Sin comentario adicional.",
                "fecha": doc.get("fecha") or "",
                "etiqueta": etiqueta,
                "clase_badge": clase_badge,
                "icono": icono,
            }
        )

    lista_calificaciones.sort(key=_ordenar_por_id, reverse=True)
    promedio = round(suma_estrellas / len(lista_calificaciones), 1) if lista_calificaciones else 0.0

    return motor_plantillas.TemplateResponse(
        solicitud,
        "calificacion/calificacion_inicio.html",
        {
            "calificaciones": lista_calificaciones,
            "promedio_general": f"{promedio:.1f}",
            "total_resenas": len(lista_calificaciones),
            "es_admin": False,
        },
    )


@router.get("/nuevo")
def vista_crear_calificacion(solicitud: Request):
    """
    Ruta HTTP GET para crear calificación:
    Si es admin, carga 'admin/calificacion_form_admin.html';
    si no, redirige al Historial de Rutas (/ruta).
    """
    if _es_admin(solicitud):
        usuarios = list(db["usuario"].find({}, {"_id": 0}))
        viajes = _enriquecer_viajes_con_ruta()

        return motor_plantillas.TemplateResponse(
            solicitud,
            "admin/calificacion_form_admin.html",
            {
                "calificacion": None,
                "usuarios": usuarios,
                "viajes": viajes,
                "es_admin": True,
            },
        )

    return RedirectResponse(
        url="/ruta",
        status_code=status.HTTP_303_SEE_OTHER,
    )


@router.post("/guardar")
def guardar_calificacion(
    solicitud: Request,
    puntuacion: int = Form(...),
    comentario: str = Form(...),
    id_usuario: int = Form(...),
    id_viaje: int = Form(...),
):
    """Ruta HTTP POST para procesar la creación y almacenar una nueva calificación mediante el controlador."""
    datos_extra = {}
    viaje_doc = db["viaje"].find_one({"id_viaje": id_viaje}, {"_id": 0})
    if viaje_doc:
        id_ruta_viaje = _to_int_or_none(viaje_doc.get("id_ruta"))
        if id_ruta_viaje is not None:
            datos_extra["id_ruta"] = id_ruta_viaje

        conductor_viaje = (
            _to_int_or_none(viaje_doc.get("id_conductor"))
            or _to_int_or_none(viaje_doc.get("conductor_id"))
            or _to_int_or_none(viaje_doc.get("id_usuario"))
            or _to_int_or_none(viaje_doc.get("usuario_id"))
        )
        if conductor_viaje is not None:
            datos_extra["id_conductor"] = conductor_viaje
            datos_extra["id_usuario_conductor"] = conductor_viaje

    try:
        nueva_calificacion = CalificacionModel(
            puntuacion=puntuacion,
            comentario=comentario,
            id_usuario=id_usuario,
            id_viaje=id_viaje,
            **datos_extra,
        )
        CalificacionController.crear_calificacion(nueva_calificacion)
    except TypeError:
        nueva_calificacion = CalificacionModel(
            puntuacion=puntuacion,
            comentario=comentario,
            id_usuario=id_usuario,
            id_viaje=id_viaje,
        )
        CalificacionController.crear_calificacion(nueva_calificacion)
        if datos_extra:
            ultima = db["calificacion"].find_one(
                sort=[("id_calificacion", -1)], projection={"_id": 0, "id_calificacion": 1}
            )
            if ultima and ultima.get("id_calificacion") is not None:
                db["calificacion"].update_one(
                    {"id_calificacion": ultima["id_calificacion"]},
                    {"$set": datos_extra},
                )

    url_retorno = "/calificacion?rol=admin" if _es_admin(solicitud) else "/calificacion"
    return RedirectResponse(
        url=url_retorno,
        status_code=status.HTTP_303_SEE_OTHER,
    )


@router.get("/editar/{id_calificacion}")
def vista_editar_calificacion(solicitud: Request, id_calificacion: int):
    """Ruta HTTP GET reservada para administración: carga los datos de una calificación existente en el formulario."""
    calificacion_encontrada = CalificacionController.obtener_calificacion_por_id(id_calificacion)
    usuarios = list(db["usuario"].find({}, {"_id": 0}))
    viajes = _enriquecer_viajes_con_ruta()

    es_admin = _es_admin(solicitud)
    plantilla = "admin/calificacion_form_admin.html" if es_admin else "calificacion/calificacion_form.html"

    return motor_plantillas.TemplateResponse(
        solicitud,
        plantilla,
        {
            "calificacion": calificacion_encontrada,
            "usuarios": usuarios,
            "viajes": viajes,
            "es_admin": es_admin,
        },
    )


@router.post("/actualizar")
def actualizar_calificacion(
    solicitud: Request,
    id_calificacion: int = Form(...),
    puntuacion: int = Form(None),
    comentario: str = Form(None),
    id_usuario: int = Form(None),
    id_viaje: int = Form(None),
):
    """Ruta HTTP POST reservada para administración: procesa la actualización parcial de una calificación."""
    calificacion_actualizacion = CalificacionUpdateModel(
        puntuacion=puntuacion,
        comentario=comentario,
        id_usuario=id_usuario,
        id_viaje=id_viaje,
    )

    CalificacionController.actualizar_calificacion(id_calificacion, calificacion_actualizacion)

    url_retorno = "/calificacion?rol=admin" if _es_admin(solicitud) else "/calificacion"
    return RedirectResponse(
        url=url_retorno,
        status_code=status.HTTP_303_SEE_OTHER,
    )


@router.get("/eliminar/{id_calificacion}")
def eliminar_calificacion(solicitud: Request, id_calificacion: int):
    """Ruta HTTP GET reservada para moderación del administrador: elimina definitivamente una calificación."""
    CalificacionController.eliminar_calificacion(id_calificacion)

    url_retorno = "/calificacion?rol=admin" if _es_admin(solicitud) else "/calificacion"
    return RedirectResponse(
        url=url_retorno,
        status_code=status.HTTP_303_SEE_OTHER,
    )