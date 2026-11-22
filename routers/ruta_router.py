# Importamos math para el cálculo geodésico (Haversine) y datetime para sellos de tiempo
import math
from datetime import datetime

# Importamos la instancia de base de datos MongoDB configurada en database.py
from database import db

# Importamos las herramientas de FastAPI para definir rutas, formularios y redirecciones
from fastapi import APIRouter, Form, Request, status
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates

# Inicializamos el enrutador del módulo de Rutas con prefijo '/ruta'
router = APIRouter(prefix="/ruta", tags=["Ruta"])

# Configuramos el motor de plantillas Jinja2 apuntando a la carpeta 'templates'
motor_plantillas = Jinja2Templates(directory="templates")

# Catálogo de coordenadas (latitud, longitud) de los nodos del Valle de Toluca
COORDENADAS_NODOS = {
    "Rectoría UTVT": [19.3451, -99.5082],
    "UTVT": [19.3451, -99.5082],
    "Terminal Toluca": [19.2848, -99.6552],
    "Terminal": [19.2848, -99.6552],
    "San Mateo Atenco": [19.2673, -99.5931],
    "San Mateo": [19.2673, -99.5931],
    "Toluca (Centro)": [19.2926, -99.6569],
    "Metepec": [19.2568, -99.6047],
    "Lerma": [19.2875, -99.5186],
    "Ocoyoacac": [19.2725, -99.4614],
    "Zinacantepec": [19.2811, -99.7275],
    "Xonacatlán": [19.3920, -99.5350],
    "Xonacatlan": [19.3920, -99.5350],
    "Villa Cuauhtémoc": [19.4167, -99.5667],
    "Villa Cuauhtemoc": [19.4167, -99.5667],
}


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


def _obtener_id_usuario_efectivo(solicitud: Request) -> int | None:
    """Obtiene el ID del usuario logueado o un usuario no-admin por defecto si no viene en la petición."""
    id_sesion = _obtener_id_usuario_sesion(solicitud)
    if id_sesion is not None and id_sesion > 0:
        return id_sesion

    usuario_no_admin = db["usuario"].find_one(
        {"rol": {"$not": {"$regex": "admin", "$options": "i"}}},
        {"_id": 0, "id_usuario": 1},
    )
    if usuario_no_admin and usuario_no_admin.get("id_usuario") is not None:
        raw_id = str(usuario_no_admin.get("id_usuario")).strip()
        if raw_id.isdigit():
            return int(raw_id)

    return None


def _obtener_usuario_por_id_o_nombre(id_usuario: int, nombre_usuario: str = "") -> dict | None:
    """Busca un usuario real en db['usuario'] por id_usuario o nombre."""
    if id_usuario > 0:
        doc = db["usuario"].find_one({"id_usuario": id_usuario}, {"_id": 0})
        if doc:
            return doc
    if nombre_usuario:
        return db["usuario"].find_one(
            {"nombre": {"$regex": f"^{nombre_usuario}$", "$options": "i"}},
            {"_id": 0},
        )
    return None


def _pertenece_a_usuario(
    doc: dict,
    id_usuario_activo: int | None,
    usuario_sesion: dict | None = None,
) -> bool:
    """Verifica estrictamente si una ruta pertenece al usuario activo cuando no es admin."""
    if id_usuario_activo is None:
        return False

    campos_id = [
        doc.get("id_usuario"),
        doc.get("id_conductor"),
        doc.get("usuario_id"),
        doc.get("conductor_id"),
    ]
    ids_presentes = [int(v) for v in campos_id if v is not None and str(v).strip().isdigit()]
    if id_usuario_activo in ids_presentes:
        return True

    if not ids_presentes and usuario_sesion:
        nombre_sesion = (
            f"{usuario_sesion.get('nombre', '')} {usuario_sesion.get('apellido_paterno', '')}".strip().lower()
        )
        nombre_doc = str(
            doc.get("nombre_usuario") or doc.get("nombre_pasajero") or doc.get("pasajero") or ""
        ).strip().lower()
        if nombre_sesion and nombre_doc and nombre_sesion == nombre_doc:
            return True

    return False


def _estimar_metricas(origen: str, destino: str) -> tuple[float, int, float]:
    """Calcula distancia vial aproximada (km), duración (min) y tarifa cuando una ruta viene en 0."""
    c1 = COORDENADAS_NODOS.get(origen, [19.2848, -99.6552])
    c2 = COORDENADAS_NODOS.get(destino, [19.3451, -99.5082])

    lat1, lon1 = math.radians(c1[0]), math.radians(c1[1])
    lat2, lon2 = math.radians(c2[0]), math.radians(c2[1])
    a = math.sin((lat2 - lat1) / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin((lon2 - lon1) / 2) ** 2

    km = round(max(4.5, 6371.0 * (2 * math.asin(math.sqrt(a))) * 1.22), 1)
    minutos = max(12, int(round(km * 2.1)))
    costo = 20.0 if km <= 12.0 else 25.0
    return km, minutos, costo


def _obtener_rutas_catalogadas() -> list[dict]:
    """Consulta las rutas existentes en db['ruta'] para alimentar el selector de rutas."""
    rutas_raw = list(db["ruta"].find({}, {"_id": 0}))
    rutas_limpias = []
    vistos = set()

    for idx, r in enumerate(rutas_raw):
        origen_r = str(r.get("origen") or r.get("punto_partida") or r.get("punto_origen") or "").strip()
        destino_r = str(r.get("destino") or r.get("punto_llegada") or r.get("punto_destino") or "").strip()
        if not origen_r and not destino_r:
            continue

        id_r = r.get("id_ruta") if r.get("id_ruta") is not None else (idx + 1)
        clave = (origen_r.lower(), destino_r.lower())
        if clave in vistos:
            continue
        vistos.add(clave)

        rutas_limpias.append(
            {
                "id_ruta": id_r,
                "origen": origen_r,
                "destino": destino_r,
            }
        )

    return rutas_limpias


def _recalcular_calificacion_pasajero(id_usuario: int, nombre_pasajero: str):
    """Recalcula el promedio real del pasajero en db['usuario'] cuando el conductor lo califica."""
    filtros = []
    if id_usuario > 0:
        filtros.append({"id_usuario": id_usuario})
    if nombre_pasajero:
        filtros.append({"nombre_pasajero": {"$regex": f"^{nombre_pasajero}$", "$options": "i"}})

    if not filtros:
        return

    resenas = list(db["calificacion_pasajero"].find({"$or": filtros}, {"_id": 0}))
    valores = [int(r["estrellas"]) for r in resenas if r.get("estrellas") is not None]
    if not valores:
        return

    promedio_conductor = sum(valores) / len(valores)

    usuario = db["usuario"].find_one({"id_usuario": id_usuario}) if id_usuario > 0 else None
    if not usuario and nombre_pasajero:
        usuario = db["usuario"].find_one({"nombre": {"$regex": f"^{nombre_pasajero}$", "$options": "i"}})

    if usuario:
        cancelaciones = int(usuario.get("cancelaciones", 0))
        calificacion_final = max(1.0, min(5.0, round(promedio_conductor - (cancelaciones * 0.15), 1)))
        db["usuario"].update_one(
            {"_id": usuario["_id"]},
            {"$set": {"calificacion": calificacion_final}},
        )


@router.get("/", response_class=HTMLResponse)
def listar_rutas(solicitud: Request):
    """Ruta HTTP GET que muestra únicamente las rutas publicadas en 'ruta' según el rol y usuario."""
    es_admin = _es_admin(solicitud)
    id_usuario_activo = None if es_admin else _obtener_id_usuario_efectivo(solicitud)
    plantilla = "admin/ruta_inicio_admin.html" if es_admin else "ruta/ruta_inicio.html"

    usuario_sesion = (
        _obtener_usuario_por_id_o_nombre(id_usuario_activo or 0)
        if id_usuario_activo
        else None
    )

    tarjetas = []
    rutas_raw = list(db["ruta"].find({}, {"_id": 0}))

    for idx, r in enumerate(rutas_raw):
        if not es_admin and not _pertenece_a_usuario(r, id_usuario_activo, usuario_sesion):
            continue

        id_ruta = int(r.get("id_ruta", idx + 1))
        id_u = int(r.get("id_usuario") or (id_usuario_activo or 0))
        usuario_doc = _obtener_usuario_por_id_o_nombre(
            id_u,
            str(r.get("nombre_usuario") or r.get("nombre_pasajero") or r.get("pasajero") or ""),
        ) or (usuario_sesion if not es_admin else None)

        nombre_usuario_doc = (
            f"{usuario_doc.get('nombre', '')} {usuario_doc.get('apellido_paterno', '')}".strip()
            if usuario_doc
            else ""
        )
        nombre_p = str(
            r.get("nombre_usuario")
            or r.get("nombre_pasajero")
            or r.get("pasajero")
            or nombre_usuario_doc
            or "Conductor"
        ).strip()
        matricula = str(
            r.get("matricula")
            or (usuario_doc.get("matricula") if usuario_doc and usuario_doc.get("matricula") else "")
        ).strip()

        origen = r.get("origen") or r.get("punto_partida") or "Origen"
        destino = r.get("destino") or r.get("punto_llegada") or "Destino"

        km_est, min_est, costo_est = _estimar_metricas(origen, destino)
        distancia = float(r.get("distancia_km") or r.get("distancia") or 0)
        tiempo = int(r.get("tiempo_min") or r.get("duracion_estimada") or 0)
        costo = float(r.get("costo") or 0)
        lugares = int(r.get("lugares") or r.get("asientos_disponibles") or 0)

        if distancia <= 0:
            distancia = km_est
        if tiempo <= 0:
            tiempo = min_est
        if costo <= 0:
            costo = costo_est
        if lugares <= 0:
            lugares = 1

        cal_doc = db["calificacion_pasajero"].find_one({"origen_tipo": "ruta", "id_registro": id_ruta}, {"_id": 0})
        tarjetas.append(
            {
                "id_ruta": id_ruta,
                "origen_tipo": "ruta",
                "id_registro": id_ruta,
                "id_usuario": id_u,
                "nombre_pasajero": nombre_p,
                "nombre_usuario": str(r.get("nombre_usuario") or nombre_usuario_doc or nombre_p),
                "hora_salida": str(r.get("hora_salida") or ""),
                "inicial": nombre_p[0].upper() if nombre_p else "C",
                "matricula": matricula,
                "origen": origen,
                "destino": destino,
                "distancia_km": distancia,
                "tiempo_min": tiempo,
                "costo": int(costo) if costo.is_integer() else costo,
                "lugares": lugares,
                "calificado": bool(cal_doc),
                "estrellas_dadas": cal_doc.get("estrellas") if cal_doc else None,
                "comentario_dado": cal_doc.get("comentario") if cal_doc else "",
                "puede_eliminar": True,
            }
        )

    return motor_plantillas.TemplateResponse(
        solicitud,
        plantilla,
        {
            "rutas": tarjetas,
            "es_admin": es_admin,
        },
    )


@router.get("/nuevo", response_class=HTMLResponse)
def vista_crear_ruta(solicitud: Request):
    """Ruta HTTP GET para cargar el formulario vacío destinado a registrar una nueva ruta."""
    usuarios = list(db["usuario"].find({}, {"_id": 0}))
    rutas_disponibles = _obtener_rutas_catalogadas()
    es_admin = _es_admin(solicitud)
    plantilla = "admin/ruta_form_admin.html" if es_admin else "ruta/ruta_form.html"

    return motor_plantillas.TemplateResponse(
        solicitud,
        plantilla,
        {
            "ruta": None,
            "rutas": rutas_disponibles,
            "rutas_disponibles": rutas_disponibles,
            "usuarios": usuarios,
            "es_admin": es_admin,
        },
    )


@router.post("/guardar")
def guardar_ruta(
    solicitud: Request,
    origen: str = Form(...),
    destino: str = Form(...),
    hora_salida: str = Form(""),
    distancia_km: float = Form(0.0),
    tiempo_min: int = Form(0),
    costo: float = Form(25.0),
    id_usuario: int = Form(0),
):
    """Ruta HTTP POST para procesar el formulario de creación y almacenar la nueva ruta."""
    ultima = db["ruta"].find_one(sort=[("id_ruta", -1)])
    nuevo_id = (ultima["id_ruta"] + 1) if ultima and "id_ruta" in ultima else 1

    id_usuario_final = id_usuario if id_usuario > 0 else (_obtener_id_usuario_efectivo(solicitud) or 0)
    usuario_doc = _obtener_usuario_por_id_o_nombre(id_usuario_final) if id_usuario_final > 0 else None
    nombre_creador = (
        f"{usuario_doc.get('nombre', '')} {usuario_doc.get('apellido_paterno', '')}".strip()
        if usuario_doc
        else ""
    )
    matricula_creador = str(usuario_doc.get("matricula") or "").strip() if usuario_doc else ""

    km_est, min_est, costo_est = _estimar_metricas(origen, destino)
    distancia_final = distancia_km if distancia_km and distancia_km > 0 else km_est
    tiempo_final = tiempo_min if tiempo_min and tiempo_min > 0 else min_est
    costo_final = costo if costo and costo > 0 else costo_est

    db["ruta"].insert_one(
        {
            "id_ruta": nuevo_id,
            "id_usuario": id_usuario_final,
            "nombre_usuario": nombre_creador,
            "nombre_pasajero": nombre_creador,
            "matricula": matricula_creador,
            "origen": origen,
            "destino": destino,
            "hora_salida": hora_salida,
            "distancia_km": distancia_final,
            "tiempo_min": tiempo_final,
            "costo": costo_final,
            "lugares": 3,
        }
    )

    url_retorno = "/ruta?rol=admin" if _es_admin(solicitud) else "/ruta"
    return RedirectResponse(
        url=url_retorno,
        status_code=status.HTTP_303_SEE_OTHER,
    )


@router.get("/editar/{id_ruta}", response_class=HTMLResponse)
def vista_editar_ruta(solicitud: Request, id_ruta: int):
    """Ruta HTTP GET para cargar los datos de una ruta existente dentro del formulario de edición."""
    ruta_encontrada = db["ruta"].find_one({"id_ruta": id_ruta}, {"_id": 0})
    usuarios = list(db["usuario"].find({}, {"_id": 0}))
    rutas_disponibles = _obtener_rutas_catalogadas()
    es_admin = _es_admin(solicitud)
    plantilla = "admin/ruta_form_admin.html" if es_admin else "ruta/ruta_form.html"

    return motor_plantillas.TemplateResponse(
        solicitud,
        plantilla,
        {
            "ruta": ruta_encontrada,
            "rutas": rutas_disponibles,
            "rutas_disponibles": rutas_disponibles,
            "usuarios": usuarios,
            "es_admin": es_admin,
        },
    )


@router.post("/actualizar")
def actualizar_ruta(
    solicitud: Request,
    id_ruta: int = Form(...),
    origen: str = Form(None),
    destino: str = Form(None),
    hora_salida: str = Form(None),
    distancia_km: float = Form(None),
    tiempo_min: int = Form(None),
    costo: float = Form(None),
    id_usuario: int = Form(None),
):
    """Ruta HTTP POST para procesar la actualización de una ruta existente."""
    campos_set = {}
    if origen is not None:
        campos_set["origen"] = origen
    if destino is not None:
        campos_set["destino"] = destino
    if hora_salida is not None:
        campos_set["hora_salida"] = hora_salida
    if distancia_km is not None:
        campos_set["distancia_km"] = distancia_km
    if tiempo_min is not None:
        campos_set["tiempo_min"] = tiempo_min
    if costo is not None:
        campos_set["costo"] = costo
    if id_usuario is not None and id_usuario > 0:
        campos_set["id_usuario"] = id_usuario
        usuario_doc = _obtener_usuario_por_id_o_nombre(id_usuario)
        if usuario_doc:
            nombre_u = f"{usuario_doc.get('nombre', '')} {usuario_doc.get('apellido_paterno', '')}".strip()
            campos_set["nombre_usuario"] = nombre_u
            campos_set["nombre_pasajero"] = nombre_u
            if usuario_doc.get("matricula"):
                campos_set["matricula"] = str(usuario_doc.get("matricula")).strip()

    if campos_set:
        db["ruta"].update_one({"id_ruta": id_ruta}, {"$set": campos_set})

    url_retorno = "/ruta?rol=admin" if _es_admin(solicitud) else "/ruta"
    return RedirectResponse(
        url=url_retorno,
        status_code=status.HTTP_303_SEE_OTHER,
    )


@router.get("/eliminar/{id_ruta}")
def eliminar_ruta(solicitud: Request, id_ruta: int, origen_tipo: str = "ruta"):
    """Ruta HTTP GET para procesar la eliminación definitiva de una ruta o reserva."""
    if origen_tipo == "reserva":
        db["reserva"].delete_one({"id_reserva": id_ruta})
    else:
        db["ruta"].delete_one({"id_ruta": id_ruta})

    url_retorno = "/ruta?rol=admin" if _es_admin(solicitud) else "/ruta"
    return RedirectResponse(
        url=url_retorno,
        status_code=status.HTTP_303_SEE_OTHER,
    )


@router.post("/calificar-pasajero")
def calificar_pasajero(
    solicitud: Request,
    id_conductor: int = Form(0),
    nombre_conductor: str = Form(""),
    origen_tipo: str = Form(...),
    id_registro: int = Form(...),
    id_usuario: int = Form(0),
    nombre_pasajero: str = Form(...),
    matricula: str = Form(""),
    origen: str = Form(""),
    destino: str = Form(""),
    estrellas: int = Form(...),
    comentario: str = Form(""),
):
    """Ruta HTTP POST que guarda la calificación del conductor hacia un pasajero al finalizar el viaje."""
    estrellas_val = max(1, min(5, int(estrellas)))
    comentario_limpio = comentario.strip() or "Excelente pasajero, puntual y respetuoso durante el trayecto."
    fecha_actual = datetime.now().strftime("%Y-%m-%d %H:%M")

    conductor_limpio = nombre_conductor.strip()
    if not conductor_limpio and id_conductor > 0:
        doc_cond = _obtener_usuario_por_id_o_nombre(id_conductor)
        if doc_cond:
            conductor_limpio = f"{doc_cond.get('nombre', '')} {doc_cond.get('apellido_paterno', '')}".strip()
    if not conductor_limpio:
        conductor_limpio = "Conductor"

    ya_calificado = db["calificacion_pasajero"].find_one(
        {"origen_tipo": origen_tipo, "id_registro": id_registro}
    )

    if not ya_calificado:
        ultima = db["calificacion_pasajero"].find_one(sort=[("id_calificacion_pasajero", -1)])
        nuevo_id = (ultima["id_calificacion_pasajero"] + 1) if ultima and "id_calificacion_pasajero" in ultima else 1

        doc = {
            "id_calificacion_pasajero": nuevo_id,
            "origen_tipo": origen_tipo,
            "id_registro": id_registro,
            "id_usuario": id_usuario,
            "id_conductor": id_conductor,
            "nombre_pasajero": nombre_pasajero,
            "matricula_pasajero": matricula,
            "nombre_conductor": conductor_limpio,
            "origen": origen,
            "destino": destino,
            "estrellas": estrellas_val,
            "comentario": comentario_limpio,
            "fecha": fecha_actual,
        }
        db["calificacion_pasajero"].insert_one(doc)

        if origen_tipo == "reserva":
            db["reserva"].update_one(
                {"id_reserva": id_registro},
                {
                    "$set": {
                        "calificado_por_conductor": True,
                        "calificacion_del_conductor": estrellas_val,
                        "comentario_del_conductor": comentario_limpio,
                        "fecha_calificacion_conductor": fecha_actual,
                    }
                },
            )
        else:
            db["ruta"].update_one(
                {"id_ruta": id_registro},
                {
                    "$set": {
                        "calificado_por_conductor": True,
                        "calificacion_del_conductor": estrellas_val,
                        "comentario_del_conductor": comentario_limpio,
                    }
                },
            )

        _recalcular_calificacion_pasajero(id_usuario, nombre_pasajero)

    url_retorno = "/ruta?rol=admin" if _es_admin(solicitud) else "/ruta"
    return RedirectResponse(
        url=url_retorno,
        status_code=status.HTTP_303_SEE_OTHER,
    )