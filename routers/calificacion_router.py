# Importamos la clase controladora completa para gestionar la lógica de negocio de las calificaciones
from controllers.calificacion_controller import CalificacionController

# Importamos los modelos de Pydantic necesarios para validar los datos de la calificación
from models.calificacion_model import CalificacionModel, CalificacionUpdateModel

# Importamos la base de datos para poder consultar los usuarios, viajes, reservas y rutas registrados
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


def _to_int_or_none(valor) -> int | None:
    """Convierte a entero si es válido; de lo contrario devuelve None."""
    if valor is None:
        return None
    texto = str(valor).strip()
    if texto.isdigit():
        return int(texto)
    return None


def _es_admin(solicitud: Request) -> bool:
    """Detecta si la petición pertenece a una sesión de Administrador."""
    rol_query = (solicitud.query_params.get("rol") or "").strip().lower()
    rol_header = (solicitud.headers.get("x-velox-rol") or "").strip().lower()
    rol_cookie = (solicitud.cookies.get("velox_usuario_rol") or "").strip().lower()
    return "admin" in rol_query or "admin" in rol_header or "admin" in rol_cookie


def _obtener_id_usuario_sesion(solicitud: Request) -> int | None:
    """Obtiene el ID numérico del usuario activo dando prioridad a query, header o cookie."""
    id_query = _to_int_or_none(solicitud.query_params.get("id_usuario"))
    if id_query and id_query > 0:
        return id_query

    id_header = _to_int_or_none(solicitud.headers.get("x-velox-usuario-id"))
    if id_header and id_header > 0:
        return id_header

    id_cookie = _to_int_or_none(solicitud.cookies.get("velox_usuario_id"))
    if id_cookie and id_cookie > 0:
        return id_cookie

    return None


def _construir_usuarios_map() -> dict[str, dict]:
    """Indexa usuarios por clave string normalizada para soportar int/str y distintos campos de ID."""
    mapa: dict[str, dict] = {}
    for u in db["usuario"].find({}, {"_id": 0}):
        for campo_id in ("id_usuario", "usuario_id", "id"):
            val = u.get(campo_id)
            if val is not None and str(val).strip() != "":
                mapa[str(val).strip()] = u
    return mapa


def _buscar_usuario(id_val, usuarios_map: dict[str, dict]) -> dict | None:
    """Busca un usuario en el mapa normalizado."""
    if id_val is None:
        return None
    clave = str(id_val).strip()
    return usuarios_map.get(clave) if clave else None


def _obtener_usuario_activo(
    solicitud: Request,
    es_admin: bool,
    usuarios_map: dict[str, dict] | None = None,
) -> dict:
    """Obtiene el documento del usuario activo en sesión o por defecto según privilegios."""
    mapa = usuarios_map if usuarios_map is not None else _construir_usuarios_map()
    id_sesion = _obtener_id_usuario_sesion(solicitud)
    if id_sesion and id_sesion > 0:
        doc = _buscar_usuario(id_sesion, mapa)
        if doc:
            return doc
        return {"id_usuario": id_sesion, "nombre": f"Usuario #{id_sesion}", "rol": "Usuario"}

    rol_param = (
        solicitud.query_params.get("rol")
        or solicitud.headers.get("x-velox-rol")
        or solicitud.cookies.get("velox_usuario_rol")
        or ""
    ).strip()

    if es_admin:
        for u in mapa.values():
            if "admin" in str(u.get("rol") or "").lower():
                return u
        return {"nombre": "Administrador", "rol": rol_param or "Administrador"}

    if rol_param:
        for u in mapa.values():
            if rol_param.lower() in str(u.get("rol") or "").lower():
                return u

    conductor_def = db["usuario"].find_one(
        {"$or": [{"id_usuario": 5}, {"correo_institucional": "al222050411@utvtol.edu.mx"}]},
        {"_id": 0},
    )
    if conductor_def:
        return conductor_def

    for u in mapa.values():
        if "admin" not in str(u.get("rol") or "").lower():
            return u

    return {"nombre": "Usuario", "rol": rol_param or "Conductor"}


def _es_rol_pasajero(solicitud: Request, usuario_activo: dict) -> bool:
    """Determina si el usuario activo actúa como pasajero."""
    rol_explicito = (
        solicitud.query_params.get("rol")
        or solicitud.headers.get("x-velox-rol")
        or solicitud.cookies.get("velox_usuario_rol")
        or ""
    ).strip().lower()
    if rol_explicito:
        return "pasajero" in rol_explicito

    rol_bd = str(usuario_activo.get("rol") or "").strip().lower()
    return "pasajero" in rol_bd and "conductor" not in rol_bd


def _obtener_id_usuario_efectivo(solicitud: Request, usuario_activo: dict) -> int | None:
    """Resuelve el id_usuario numérico del usuario activo dando prioridad a query/header."""
    id_query = _to_int_or_none(solicitud.query_params.get("id_usuario"))
    if id_query and id_query > 0:
        return id_query

    id_header = _to_int_or_none(solicitud.headers.get("x-velox-usuario-id"))
    if id_header and id_header > 0:
        return id_header

    for campo in ("id_usuario", "usuario_id", "id"):
        id_doc = _to_int_or_none(usuario_activo.get(campo))
        if id_doc and id_doc > 0:
            return id_doc

    id_cookie = _to_int_or_none(solicitud.cookies.get("velox_usuario_id"))
    if id_cookie and id_cookie > 0:
        return id_cookie

    return None


def _construir_url_retorno(solicitud: Request, id_usuario_extra: int | None = None) -> str:
    """Construye la URL de retorno conservando el rol o id_usuario activo."""
    if _es_admin(solicitud):
        return "/calificacion/?rol=admin"
    id_ret = (
        id_usuario_extra
        if (id_usuario_extra is not None and id_usuario_extra > 0)
        else _obtener_id_usuario_sesion(solicitud)
    )
    if id_ret and id_ret > 0:
        return f"/calificacion/?id_usuario={id_ret}"
    return "/calificacion/"


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
            {"id_usuario": str(id_conductor)},
            {"id_conductor": str(id_conductor)},
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
        {"id_usuario": str(id_conductor)},
        {"id_conductor": str(id_conductor)},
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


def _obtener_viajes_pasajero(id_pasajero: int | None) -> set[int]:
    """Obtiene los id_viaje donde participó el pasajero según db['reserva']."""
    if not id_pasajero or id_pasajero <= 0:
        return set()

    reservas = list(
        db["reserva"].find(
            {
                "$or": [
                    {"id_usuario": id_pasajero},
                    {"id_pasajero": id_pasajero},
                    {"usuario_id": id_pasajero},
                    {"pasajero_id": id_pasajero},
                    {"id_usuario": str(id_pasajero)},
                    {"id_pasajero": str(id_pasajero)},
                ]
            },
            {"_id": 0, "id_viaje": 1},
        )
    )
    ids_viaje = set()
    for r in reservas:
        id_v = _to_int_or_none(r.get("id_viaje"))
        if id_v is not None:
            ids_viaje.add(id_v)
    return ids_viaje


def _obtener_conductor_de_viaje(id_viaje: int | None, viajes_map: dict[int, dict]) -> int | None:
    """Devuelve el id_usuario del conductor asignado a un viaje."""
    if id_viaje is None or id_viaje not in viajes_map:
        return None
    viaje_doc = viajes_map[id_viaje]
    for campo in ("id_conductor", "conductor_id", "id_usuario", "usuario_id"):
        val = _to_int_or_none(viaje_doc.get(campo))
        if val is not None:
            return val
    return None


def _pertenece_a_conductor(
    doc: dict,
    id_conductor: int | None,
    ids_viaje: set[int],
    ids_ruta: set[int],
) -> bool:
    """Determina si una reseña recibida está relacionada con el conductor activo."""
    if not id_conductor or id_conductor <= 0:
        return False

    emisor_id = _to_int_or_none(doc.get("id_usuario") or doc.get("usuario_id"))
    if emisor_id is not None and emisor_id == id_conductor:
        return False

    for campo in ("id_conductor", "id_conductor_viaje", "id_usuario_conductor", "conductor_id"):
        val_id = _to_int_or_none(doc.get(campo))
        if val_id is not None and val_id == id_conductor:
            return True

    id_v = _to_int_or_none(doc.get("id_viaje") or doc.get("viaje_id"))
    if id_v is not None and id_v in ids_viaje:
        return True

    id_r = _to_int_or_none(doc.get("id_ruta") or doc.get("ruta_id"))
    if id_r is not None and id_r in ids_ruta:
        return True

    return False


def _pertenece_a_pasajero(
    doc: dict,
    id_pasajero: int | None,
    ids_viaje_pasajero: set[int],
) -> bool:
    """Determina si una reseña fue dirigida realmente al pasajero activo."""
    if not id_pasajero or id_pasajero <= 0:
        return False

    emisor_id = _to_int_or_none(doc.get("id_usuario") or doc.get("usuario_id"))
    if emisor_id is not None and emisor_id == id_pasajero:
        return False

    for campo in ("id_pasajero", "id_usuario_calificado", "id_calificado", "pasajero_id"):
        val_id = _to_int_or_none(doc.get(campo))
        if val_id is not None and val_id == id_pasajero:
            return True

    tipo = str(doc.get("tipo") or "").strip().lower()
    id_v = _to_int_or_none(doc.get("id_viaje") or doc.get("viaje_id"))
    if tipo == "conductor_a_pasajero" and id_v is not None and id_v in ids_viaje_pasajero:
        return True

    return False


def _describir_emisor_para_pasajero(
    doc: dict,
    emisor_id: int | None,
    id_conductor_viaje: int | None,
) -> str:
    """Identifica si quien calificó al pasajero fue el conductor o un compañero de viaje."""
    tipo = str(doc.get("tipo") or "").strip().lower()
    if "conductor" in tipo:
        return "Conductor del viaje"
    if "pasajero" in tipo or "companero" in tipo or "compañero" in tipo:
        return "Compañero de viaje"
    if emisor_id is not None and id_conductor_viaje is not None and emisor_id == id_conductor_viaje:
        return "Conductor del viaje"
    return "Compañero de viaje"


def _formatear_nombre_usuario(usuario: dict) -> str:
    """Construye el nombre real de un documento de usuario."""
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
    for campo in ("nombre_completo", "usuario", "correo", "email"):
        val = str(usuario.get(campo) or "").strip()
        if val:
            return val
    return ""


def _resolver_nombre_por_id(id_u, usuarios_map: dict[str, dict]) -> str:
    """Busca el nombre real de un usuario por su ID en usuarios_map."""
    u_doc = _buscar_usuario(id_u, usuarios_map)
    if u_doc:
        nombre_real = _formatear_nombre_usuario(u_doc)
        if nombre_real:
            return nombre_real
    return ""


def _resolver_nombre_emisor(
    doc: dict, usuarios_map: dict[str, dict], fallback_rol: str = "Pasajero"
) -> str:
    """Obtiene el nombre real del emisor desde la reseña o desde db['usuario']."""
    nombre_guardado = str(
        doc.get("nombre_emisor") or doc.get("nombre_pasajero") or doc.get("pasajero") or ""
    ).strip()
    if nombre_guardado and not nombre_guardado.lower().startswith("usuario"):
        return nombre_guardado

    id_u = _to_int_or_none(doc.get("id_usuario") or doc.get("usuario_id") or doc.get("id_emisor"))
    nombre_bd = _resolver_nombre_por_id(id_u, usuarios_map)
    if nombre_bd:
        return nombre_bd

    if id_u is not None:
        return f"Usuario #{id_u}"
    return fallback_rol


def _resolver_nombre_calificado_emitida(
    doc: dict,
    id_emisor: int | None,
    id_viaje_doc: int | None,
    viajes_map: dict[int, dict],
    usuarios_map: dict[str, dict],
    nombre_usuario_activo: str,
    fallback_rol: str = "Usuario calificado",
    indice_emitida: int = 0,
) -> str:
    """Resuelve a quién calificó realmente el usuario activo sin devolver al mismo emisor."""
    nombre_activo_norm = (nombre_usuario_activo or "").strip().lower()
    id_cond_v = _obtener_conductor_de_viaje(id_viaje_doc, viajes_map)

    id_calificado_directo = _to_int_or_none(
        doc.get("id_pasajero")
        or doc.get("id_usuario_calificado")
        or doc.get("id_calificado")
        or doc.get("pasajero_id")
    )
    if id_calificado_directo == id_emisor:
        id_calificado_directo = None

    id_cond_doc = _to_int_or_none(
        doc.get("id_conductor")
        or doc.get("conductor_id")
        or doc.get("id_usuario_conductor")
    )
    if id_cond_doc == id_emisor:
        id_cond_doc = None

    candidatos_ids: list[int | None] = []
    if id_calificado_directo is not None:
        candidatos_ids.append(id_calificado_directo)

    if id_emisor is not None and id_cond_v is not None and id_emisor == id_cond_v:
        if id_viaje_doc is not None:
            reservas_viaje = list(
                db["reserva"].find(
                    {"$or": [{"id_viaje": id_viaje_doc}, {"viaje_id": id_viaje_doc}]},
                    {"_id": 0},
                )
            )
            pasajeros_viaje: list[int] = []
            for res in reservas_viaje:
                id_pas_res = _to_int_or_none(
                    res.get("id_pasajero")
                    or res.get("id_usuario")
                    or res.get("usuario_id")
                    or res.get("pasajero_id")
                )
                if id_pas_res is not None and id_pas_res != id_emisor and id_pas_res != id_cond_v:
                    if id_pas_res not in pasajeros_viaje:
                        pasajeros_viaje.append(id_pas_res)

            if pasajeros_viaje and id_calificado_directo is None:
                elegido = pasajeros_viaje[indice_emitida % len(pasajeros_viaje)]
                candidatos_ids.append(elegido)
            for p_id in pasajeros_viaje:
                if p_id not in candidatos_ids:
                    candidatos_ids.append(p_id)
    else:
        for c in (id_cond_doc, id_cond_v, id_calificado_directo):
            if c is not None and c != id_emisor and c not in candidatos_ids:
                candidatos_ids.append(c)

    for cand_id in candidatos_ids:
        if cand_id is not None and cand_id != id_emisor:
            nombre = _resolver_nombre_por_id(cand_id, usuarios_map)
            if nombre and nombre.strip().lower() != nombre_activo_norm:
                return nombre

    for campo_nombre in ("nombre_calificado", "nombre_pasajero", "nombre_conductor", "pasajero"):
        nombre_texto = str(doc.get(campo_nombre) or "").strip()
        if (
            nombre_texto
            and not nombre_texto.lower().startswith("usuario")
            and nombre_texto.lower() != nombre_activo_norm
        ):
            return nombre_texto

    for cand_id in candidatos_ids:
        if cand_id is not None and cand_id != id_emisor:
            return f"Usuario #{cand_id}"

    return fallback_rol


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


def _extraer_origen_destino(doc: dict) -> tuple[str, str]:
    """Extrae origen y destino de un documento de viaje o ruta."""
    origen = str(
        doc.get("origen")
        or doc.get("punto_partida")
        or doc.get("lugar_origen")
        or doc.get("inicio")
        or ""
    ).strip()
    destino = str(
        doc.get("destino")
        or doc.get("punto_llegada")
        or doc.get("lugar_destino")
        or doc.get("fin")
        or ""
    ).strip()
    return origen, destino


def _enriquecer_viajes_con_ruta() -> list[dict]:
    """Obtiene todos los viajes de db['viaje'] con origen y destino resueltos."""
    viajes_raw = list(db["viaje"].find({}, {"_id": 0}))
    rutas_map = {
        int(r["id_ruta"]): r
        for r in db["ruta"].find({}, {"_id": 0})
        if r.get("id_ruta") is not None and str(r.get("id_ruta")).strip().isdigit()
    }
    viajes = []
    for v in viajes_raw:
        origen, destino = _extraer_origen_destino(v)
        id_ruta = _to_int_or_none(v.get("id_ruta") or v.get("ruta_id"))
        if (not origen or not destino) and id_ruta is not None and id_ruta in rutas_map:
            r_origen, r_destino = _extraer_origen_destino(rutas_map[id_ruta])
            origen = origen or r_origen
            destino = destino or r_destino
        v["origen"] = origen or "Origen"
        v["destino"] = destino or "Destino"
        viajes.append(v)
    return viajes


def _resolver_ruta_viaje(
    doc: dict,
    id_viaje: int | None,
    viajes_map: dict[int, dict],
    rutas_map: dict[int, dict],
    id_ruta_extra: int | None = None,
) -> str:
    """Resuelve el texto 'Origen ➔ Destino' real desde el viaje, la ruta o el documento."""
    viaje_info = viajes_map.get(id_viaje or -1, {})
    origen, destino = _extraer_origen_destino(viaje_info)

    id_ruta = (
        _to_int_or_none(viaje_info.get("id_ruta") or viaje_info.get("ruta_id"))
        or id_ruta_extra
        or _to_int_or_none(doc.get("id_ruta") or doc.get("ruta_id"))
    )
    if (not origen or not destino) and id_ruta is not None and id_ruta in rutas_map:
        r_origen, r_destino = _extraer_origen_destino(rutas_map[id_ruta])
        origen = origen or r_origen
        destino = destino or r_destino

    if not origen or not destino:
        d_origen, d_destino = _extraer_origen_destino(doc)
        origen = origen or d_origen
        destino = destino or d_destino

    if origen and destino:
        return f"{origen} ➔ {destino}"
    if origen or destino:
        return origen or destino
    return ""


@router.get("/")
def listar_calificaciones(solicitud: Request):
    """Ruta HTTP GET para mostrar la vista principal de calificaciones."""
    lista_raw = CalificacionController.obtener_todas_las_calificaciones()
    usuarios_map = _construir_usuarios_map()
    es_admin = _es_admin(solicitud)
    usuario_activo = _obtener_usuario_activo(solicitud, es_admin, usuarios_map)

    if es_admin:
        calificaciones_admin = [
            item.model_dump()
            if hasattr(item, "model_dump")
            else (item.dict() if hasattr(item, "dict") else dict(item))
            for item in lista_raw
        ]
        calificaciones_admin.sort(key=_ordenar_por_id, reverse=True)
        resp = motor_plantillas.TemplateResponse(
            solicitud,
            "admin/calificacion_inicio_admin.html",
            {
                "calificaciones": calificaciones_admin,
                "calificaciones_emitidas": [],
                "total_resenas": len(calificaciones_admin),
                "es_admin": True,
                "usuario": usuario_activo,
            },
        )
        resp.set_cookie("velox_usuario_rol", "admin", path="/")
        return resp

    id_usuario = _obtener_id_usuario_efectivo(solicitud, usuario_activo)
    es_pasajero = _es_rol_pasajero(solicitud, usuario_activo)

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

    if es_pasajero:
        ids_viaje_pasajero = _obtener_viajes_pasajero(id_usuario)
        ids_viaje_cond, ids_ruta_cond = set(), set()
    else:
        ids_viaje_cond, ids_ruta_cond = _obtener_ids_viaje_y_ruta_conductor(id_usuario)
        ids_viaje_pasajero = set()

    lista_calificaciones = []
    lista_emitidas = []
    suma_estrellas = 0
    nombre_usuario_activo = _resolver_nombre_por_id(id_usuario, usuarios_map)

    for idx, item in enumerate(lista_raw):
        doc = (
            item.model_dump()
            if hasattr(item, "model_dump")
            else (item.dict() if hasattr(item, "dict") else dict(item))
        )

        estrellas = _extraer_estrellas(doc)
        id_cal = doc.get("id_calificacion", idx + 1)
        id_viaje_doc = _to_int_or_none(doc.get("id_viaje") or doc.get("viaje_id"))
        id_ruta_doc = _to_int_or_none(doc.get("id_ruta") or doc.get("ruta_id"))
        emisor_id = _to_int_or_none(doc.get("id_usuario") or doc.get("usuario_id"))
        ruta_texto = _resolver_ruta_viaje(doc, id_viaje_doc, viajes_map, rutas_map, id_ruta_doc)

        if id_usuario is not None and emisor_id == id_usuario:
            fallback_emitida = "Conductor del viaje" if es_pasajero else "Pasajero del viaje"
            nombre_calificado = _resolver_nombre_calificado_emitida(
                doc,
                emisor_id,
                id_viaje_doc,
                viajes_map,
                usuarios_map,
                nombre_usuario_activo,
                fallback_rol=fallback_emitida,
                indice_emitida=len(lista_emitidas),
            )

            lista_emitidas.append(
                {
                    "id_calificacion": id_cal,
                    "nombre_pasajero": nombre_calificado,
                    "nombre_usuario": nombre_calificado,
                    "descripcion_viaje": ruta_texto,
                    "ruta": ruta_texto,
                    "id_viaje": id_viaje_doc,
                    "inicial": nombre_calificado[0].upper() if nombre_calificado else "U",
                    "estrellas": estrellas,
                    "comentario": doc.get("comentario") or "",
                    "fecha": doc.get("fecha") or "",
                }
            )

        aplica_recibida = False
        if es_pasajero:
            aplica_recibida = _pertenece_a_pasajero(doc, id_usuario, ids_viaje_pasajero)
        else:
            if doc.get("tipo") != "conductor_a_pasajero":
                aplica_recibida = _pertenece_a_conductor(doc, id_usuario, ids_viaje_cond, ids_ruta_cond)

        if not aplica_recibida:
            continue

        suma_estrellas += estrellas
        nombre_emisor = _resolver_nombre_emisor(
            doc,
            usuarios_map,
            fallback_rol="Compañero de viaje" if es_pasajero else "Pasajero",
        )
        id_conductor_viaje = _obtener_conductor_de_viaje(id_viaje_doc, viajes_map)
        origen_resena = (
            _describir_emisor_para_pasajero(doc, emisor_id, id_conductor_viaje)
            if es_pasajero
            else "Pasajero"
        )

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
                "nombre_pasajero": nombre_emisor,
                "rol_emisor": origen_resena,
                "descripcion_viaje": ruta_texto,
                "ruta": ruta_texto,
                "id_viaje": id_viaje_doc,
                "inicial": nombre_emisor[0].upper() if nombre_emisor else "P",
                "estrellas": estrellas,
                "comentario": doc.get("comentario") or "Sin comentario adicional.",
                "fecha": doc.get("fecha") or "",
                "etiqueta": etiqueta,
                "clase_badge": clase_badge,
                "icono": icono,
            }
        )

    lista_calificaciones.sort(key=_ordenar_por_id, reverse=True)
    lista_emitidas.sort(key=_ordenar_por_id, reverse=True)
    promedio = round(suma_estrellas / len(lista_calificaciones), 1) if lista_calificaciones else 0.0

    resp = motor_plantillas.TemplateResponse(
        solicitud,
        "calificacion/calificacion_inicio.html",
        {
            "calificaciones": lista_calificaciones,
            "calificaciones_emitidas": lista_emitidas,
            "promedio_general": f"{promedio:.1f}",
            "total_resenas": len(lista_calificaciones),
            "es_admin": False,
            "es_pasajero": es_pasajero,
            "usuario": usuario_activo,
        },
    )
    if id_usuario and id_usuario > 0:
        resp.set_cookie("velox_usuario_id", str(id_usuario), path="/")
    else:
        resp.delete_cookie("velox_usuario_id", path="/")
    return resp


@router.get("/nuevo")
def vista_crear_calificacion(solicitud: Request):
    """
    Ruta HTTP GET para crear calificación:
    Si es admin, carga 'admin/calificacion_form_admin.html';
    si es usuario normal, carga 'calificacion/calificacion_form.html'.
    """
    es_admin = _es_admin(solicitud)
    usuarios = list(db["usuario"].find({}, {"_id": 0}))
    viajes = _enriquecer_viajes_con_ruta()
    usuario_activo = _obtener_usuario_activo(solicitud, es_admin)

    id_viaje_q = _to_int_or_none(solicitud.query_params.get("id_viaje"))
    id_pasajero_q = _to_int_or_none(solicitud.query_params.get("id_pasajero"))
    id_conductor_q = _to_int_or_none(solicitud.query_params.get("id_conductor"))

    plantilla = "admin/calificacion_form_admin.html" if es_admin else "calificacion/calificacion_form.html"

    resp = motor_plantillas.TemplateResponse(
        solicitud,
        plantilla,
        {
            "calificacion": None,
            "usuarios": usuarios,
            "viajes": viajes,
            "id_viaje_preseleccionado": id_viaje_q,
            "id_pasajero_preseleccionado": id_pasajero_q,
            "id_conductor_preseleccionado": id_conductor_q,
            "es_admin": es_admin,
            "usuario": usuario_activo,
        },
    )
    id_u = _obtener_id_usuario_efectivo(solicitud, usuario_activo)
    if id_u and id_u > 0:
        resp.set_cookie("velox_usuario_id", str(id_u), path="/")
    return resp


@router.post("/guardar")
def guardar_calificacion(
    solicitud: Request,
    puntuacion: int = Form(...),
    comentario: str = Form(...),
    id_usuario: int = Form(None),
    id_viaje: int = Form(...),
    id_pasajero: int = Form(None),
    id_conductor: int = Form(None),
    tipo: str = Form(None),
):
    """Ruta HTTP POST para procesar la creación y almacenar una nueva calificación mediante el controlador."""
    id_emisor = _to_int_or_none(id_usuario) or _obtener_id_usuario_sesion(solicitud)
    if not id_emisor or id_emisor <= 0:
        usuario_activo = _obtener_usuario_activo(solicitud, _es_admin(solicitud))
        id_emisor = _obtener_id_usuario_efectivo(solicitud, usuario_activo) or 5

    datos_extra = {"id_usuario": id_emisor}
    if id_pasajero is not None and id_pasajero > 0 and id_pasajero != id_emisor:
        datos_extra["id_pasajero"] = id_pasajero
        datos_extra["id_usuario_calificado"] = id_pasajero
    if tipo:
        datos_extra["tipo"] = tipo

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
            or _to_int_or_none(id_conductor)
        )
        if conductor_viaje is not None and conductor_viaje != id_emisor:
            datos_extra["id_conductor"] = conductor_viaje
            datos_extra["id_usuario_conductor"] = conductor_viaje
    elif id_conductor is not None and id_conductor > 0 and id_conductor != id_emisor:
        datos_extra["id_conductor"] = id_conductor
        datos_extra["id_usuario_conductor"] = id_conductor

    try:
        nueva_calificacion = CalificacionModel(
            puntuacion=puntuacion,
            comentario=comentario,
            id_usuario=id_emisor,
            id_viaje=id_viaje,
            **datos_extra,
        )
        CalificacionController.crear_calificacion(nueva_calificacion)
    except TypeError:
        nueva_calificacion = CalificacionModel(
            puntuacion=puntuacion,
            comentario=comentario,
            id_usuario=id_emisor,
            id_viaje=id_viaje,
        )
        CalificacionController.crear_calificacion(nueva_calificacion)

    ultima = db["calificacion"].find_one(
        sort=[("id_calificacion", -1)], projection={"_id": 0, "id_calificacion": 1}
    )
    if ultima and ultima.get("id_calificacion") is not None and datos_extra:
        db["calificacion"].update_one(
            {"id_calificacion": ultima["id_calificacion"]},
            {"$set": datos_extra},
        )

    url_retorno = _construir_url_retorno(solicitud, id_emisor)
    resp = RedirectResponse(
        url=url_retorno,
        status_code=status.HTTP_303_SEE_OTHER,
    )
    resp.set_cookie("velox_usuario_id", str(id_emisor), path="/")
    return resp


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

    url_retorno = _construir_url_retorno(solicitud, id_usuario)
    return RedirectResponse(
        url=url_retorno,
        status_code=status.HTTP_303_SEE_OTHER,
    )


@router.get("/eliminar/{id_calificacion}")
def eliminar_calificacion(solicitud: Request, id_calificacion: int):
    """Ruta HTTP GET reservada para moderación del administrador: elimina definitivamente una calificación."""
    CalificacionController.eliminar_calificacion(id_calificacion)

    url_retorno = _construir_url_retorno(solicitud)
    return RedirectResponse(
        url=url_retorno,
        status_code=status.HTTP_303_SEE_OTHER,
    )