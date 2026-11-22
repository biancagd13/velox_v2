import math

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

ESTADOS_RESERVA_A_BORDO = {
    "aceptada",
    "confirmada",
    "en_curso",
    "finalizada",
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


def _obtener_id_conductor_actual(solicitud: Request | None = None) -> int:
    """Obtiene el id_usuario del conductor en sesión o el de respaldo desde MongoDB."""
    if solicitud is not None:
        id_sesion = _obtener_id_usuario_sesion(solicitud)
        if id_sesion and id_sesion > 0:
            return id_sesion

    conductor = db["usuario"].find_one(
        {"correo_institucional": "al22288@utvt.edu.mx"},
        {"_id": 0, "id_usuario": 1},
    )
    if conductor and conductor.get("id_usuario") is not None:
        return int(conductor["id_usuario"])

    return 1


def _obtener_conductores() -> list[dict]:
    """Obtiene los usuarios desde MongoDB formateados para el selector de Conductor."""
    usuarios_raw = list(db["usuario"].find({}, {"_id": 0}))
    conductores = []

    for idx, u in enumerate(usuarios_raw):
        id_u = u.get("id_usuario")
        if id_u is None:
            id_u = u.get("id") if u.get("id") is not None else (idx + 1)

        nombre = str(u.get("nombre") or "").strip()
        ap_pat = str(u.get("apellido_paterno") or "").strip()
        ap_mat = str(u.get("apellido_materno") or "").strip()
        nombre_completo = " ".join(p for p in [nombre, ap_pat, ap_mat] if p) or str(
            u.get("nombre_completo") or u.get("correo_institucional") or f"Usuario #{id_u}"
        ).strip()

        correo = str(u.get("correo_institucional") or u.get("correo") or "").strip()
        rol = str(u.get("rol") or u.get("tipo_usuario") or "").strip()

        conductores.append(
            {
                "id_usuario": int(id_u) if str(id_u).isdigit() else id_u,
                "nombre": nombre or nombre_completo,
                "nombre_completo": nombre_completo,
                "correo_institucional": correo,
                "rol": rol,
            }
        )

    conductores.sort(key=lambda item: item["nombre_completo"].lower())
    return conductores


def _pertenece_a_conductor(doc: dict, id_usuario_activo: int | None) -> bool:
    """Verifica si el viaje o su ruta pertenece al conductor activo cuando no es admin."""
    if id_usuario_activo is None:
        return True
    campos = [
        doc.get("id_usuario"),
        doc.get("id_conductor"),
        doc.get("usuario_id"),
        doc.get("conductor_id"),
    ]
    ids_presentes = [int(v) for v in campos if v is not None and str(v).strip().isdigit()]
    if not ids_presentes:
        return False
    return id_usuario_activo in ids_presentes


def _estimar_metricas(origen: str, destino: str) -> tuple[float, int]:
    """Calcula distancia vial aproximada (km) y duración (min) cuando faltan en la ruta."""
    c1 = COORDENADAS_NODOS.get(origen, [19.2848, -99.6552])
    c2 = COORDENADAS_NODOS.get(destino, [19.3451, -99.5082])

    lat1, lon1 = math.radians(c1[0]), math.radians(c1[1])
    lat2, lon2 = math.radians(c2[0]), math.radians(c2[1])
    a = math.sin((lat2 - lat1) / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin((lon2 - lon1) / 2) ** 2

    km = round(max(4.5, 6371.0 * (2 * math.asin(math.sqrt(a))) * 1.22), 1)
    minutos = max(12, int(round(km * 2.1)))
    return km, minutos


def _to_int_or_none(valor) -> int | None:
    if valor is None:
        return None
    texto = str(valor).strip()
    if texto.isdigit():
        return int(texto)
    return None


@router.get("/")
def listar_viajes(solicitud: Request):
    """Ruta HTTP GET para mostrar la vista principal con el listado de viajes enriquecidos."""
    es_admin = _es_admin(solicitud)
    id_usuario_activo = None if es_admin else _obtener_id_usuario_sesion(solicitud)
    plantilla = "admin/viaje_inicio_admin.html" if es_admin else "viaje/viaje_inicio.html"

    viajes_raw = ViajeController.obtener_todos_los_viajes() or []
    rutas_map = {
        int(r["id_ruta"]): r
        for r in db["ruta"].find({}, {"_id": 0})
        if r.get("id_ruta") is not None
    }
    usuarios_map = {
        int(u["id_usuario"]): u
        for u in db["usuario"].find({}, {"_id": 0})
        if u.get("id_usuario") is not None
    }
    reservas_confirmadas = [
        res
        for res in db["reserva"].find({}, {"_id": 0})
        if str(res.get("estado") or "").strip().lower() in ESTADOS_RESERVA_A_BORDO
    ]

    viajes_enriquecidos = []
    for idx, v in enumerate(viajes_raw):
        id_ruta = int(v.get("id_ruta") or 0)
        ruta_doc = rutas_map.get(id_ruta, {})

        # Si el viaje no trae id_usuario, heredamos el de su ruta si existe
        doc_evaluacion = dict(v)
        if not doc_evaluacion.get("id_usuario") and ruta_doc.get("id_usuario"):
            doc_evaluacion["id_usuario"] = ruta_doc.get("id_usuario")

        if not _pertenece_a_conductor(doc_evaluacion, id_usuario_activo):
            continue

        id_viaje = int(v.get("id_viaje") or (idx + 1))
        origen = str(v.get("origen") or ruta_doc.get("origen") or "Origen").strip()
        destino = str(v.get("destino") or ruta_doc.get("destino") or "Destino").strip()
        hora_salida = str(v.get("hora_salida") or ruta_doc.get("hora_salida") or "").strip()
        fecha_viaje = str(v.get("fecha") or "").strip()

        if fecha_viaje and hora_salida:
            horario_texto = f"{fecha_viaje} · {hora_salida} hrs"
        elif hora_salida:
            horario_texto = f"Salida: {hora_salida} hrs"
        elif fecha_viaje:
            horario_texto = fecha_viaje
        else:
            horario_texto = "Horario por confirmar"

        estado_raw = str(v.get("estado") or EstadoViaje.PENDIENTE.value).strip().lower()
        en_curso = estado_raw in {"en_curso", "iniciado", "comenzado"}
        finalizado = estado_raw in {"finalizado", "finalizada", "completado"}

        if finalizado:
            estado_texto = "Viaje Finalizado"
            siguiente_estado = "en_curso"
            texto_boton = "Reiniciar Viaje"
            icono_boton = "bi-arrow-repeat"
            mensaje_confirmacion = "Se volverá a poner el viaje en curso."
        elif en_curso:
            estado_texto = "Viaje en Curso"
            siguiente_estado = "finalizado"
            texto_boton = "Finalizar Viaje"
            icono_boton = "bi-flag-checkered"
            mensaje_confirmacion = "Se marcará el viaje como finalizado y se actualizarán sus reservas."
        else:
            estado_texto = "Viaje Activo"
            siguiente_estado = "en_curso"
            texto_boton = "Comenzar Viaje"
            icono_boton = "bi-car-front-fill"
            mensaje_confirmacion = "Se iniciará el viaje con los pasajeros confirmados."

        km_est, min_est = _estimar_metricas(origen, destino)
        distancia_km = float(v.get("distancia_km") or ruta_doc.get("distancia_km") or km_est)
        tiempo_min = int(v.get("tiempo_min") or ruta_doc.get("tiempo_min") or min_est)
        costo_num = float(v.get("costo") if v.get("costo") is not None else (ruta_doc.get("costo") or 25))
        capacidad_lugares = int(
            v.get("capacidad_lugares")
            or v.get("asientos_disponibles")
            or ruta_doc.get("lugares")
            or 4
        )

        pasajeros_lista = []
        for res in reservas_confirmadas:
            res_id_viaje = _to_int_or_none(res.get("id_viaje"))
            res_id_ruta = _to_int_or_none(res.get("id_ruta"))

            mismo_viaje = res_id_viaje is not None and res_id_viaje == id_viaje
            misma_ruta = id_ruta > 0 and res_id_ruta is not None and res_id_ruta == id_ruta
            mismo_trayecto = (
                bool(str(res.get("origen") or "").strip())
                and bool(str(res.get("destino") or "").strip())
                and str(res.get("origen") or "").strip().lower() == origen.lower()
                and str(res.get("destino") or "").strip().lower() == destino.lower()
            )

            if not (mismo_viaje or misma_ruta or mismo_trayecto):
                continue

            id_pas = _to_int_or_none(res.get("id_usuario") or res.get("id_pasajero")) or 0
            doc_pas = usuarios_map.get(id_pas)
            nombre_pas = str(
                res.get("nombre_pasajero")
                or (
                    f"{doc_pas.get('nombre', '')} {doc_pas.get('apellido_paterno', '')}".strip()
                    if doc_pas
                    else ""
                )
                or (f"Usuario #{id_pas}" if id_pas > 0 else "Pasajero")
            ).strip()

            estado_res = str(res.get("estado") or "aceptada").strip().lower()
            estado_pas_texto = "Finalizada" if estado_res == "finalizada" else "Confirmado"

            pasajeros_lista.append(
                {
                    "id_usuario": id_pas,
                    "nombre": nombre_pas,
                    "inicial": nombre_pas[0].upper() if nombre_pas else "P",
                    "punto_encuentro": str(res.get("punto_encuentro") or origen).strip(),
                    "estado_texto": estado_pas_texto,
                }
            )

        viaje_item = dict(v)
        viaje_item.update(
            {
                "id_viaje": id_viaje,
                "id_ruta": id_ruta,
                "origen": origen,
                "destino": destino,
                "horario_texto": horario_texto,
                "estado_texto": estado_texto,
                "siguiente_estado": siguiente_estado,
                "texto_boton": texto_boton,
                "icono_boton": icono_boton,
                "mensaje_confirmacion": mensaje_confirmacion,
                "distancia_km": distancia_km,
                "tiempo_min": tiempo_min,
                "costo": int(costo_num) if costo_num.is_integer() else costo_num,
                "capacidad_lugares": capacidad_lugares,
                "pasajeros": pasajeros_lista,
            }
        )
        viajes_enriquecidos.append(viaje_item)

    return motor_plantillas.TemplateResponse(
        solicitud,
        plantilla,
        {
            "viajes": viajes_enriquecidos,
            "es_admin": es_admin,
        },
    )


@router.get("/nuevo")
def vista_crear_viaje(solicitud: Request):
    """Ruta HTTP GET para cargar el formulario vacío destinado a registrar un nuevo viaje."""
    id_conductor_actual = _obtener_id_conductor_actual(solicitud)
    conductores = _obtener_conductores()

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
            "conductores": conductores,
            "usuarios": conductores,
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
    conductor_id = id_usuario if id_usuario is not None and id_usuario > 0 else _obtener_id_conductor_actual(solicitud)
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


@router.post("/estado")
def cambiar_estado_viaje(
    solicitud: Request,
    id_viaje: int = Form(...),
    id_ruta: int = Form(0),
    nuevo_estado: str = Form(...),
):
    """Ruta HTTP POST para comenzar o finalizar un viaje y sincronizar reservas."""
    estado_limpio = nuevo_estado.strip().lower()
    if estado_limpio in {"en_curso", "finalizado"}:
        ViajeController.actualizar_viaje(id_viaje, ViajeUpdateModel(estado=estado_limpio))
        if estado_limpio == "finalizado":
            filtro_reservas = {
                "estado": {"$in": ["aceptada", "confirmada", "en_curso", "Aceptada", "Confirmada"]},
                "$or": [{"id_viaje": id_viaje}],
            }
            if id_ruta > 0:
                filtro_reservas["$or"].append({"id_ruta": id_ruta})
            db["reserva"].update_many(
                filtro_reservas,
                {"$set": {"estado": "finalizada"}},
            )

    url_retorno = "/viaje?rol=admin" if _es_admin(solicitud) else "/viaje"
    return RedirectResponse(
        url=url_retorno,
        status_code=status.HTTP_303_SEE_OTHER,
    )


@router.get("/editar/{id_viaje}")
def vista_editar_viaje(solicitud: Request, id_viaje: int):
    """Ruta HTTP GET para cargar los datos de un viaje existente dentro del formulario de edición."""
    viaje_encontrado = ViajeController.obtener_viaje_por_id(id_viaje)
    id_conductor_actual = _obtener_id_conductor_actual(solicitud)
    conductores = _obtener_conductores()

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
            "conductores": conductores,
            "usuarios": conductores,
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