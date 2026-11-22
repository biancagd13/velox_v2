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

# Diccionario de respaldo para asociar IDs históricos con nombres reales de pasajeros
MAPA_NOMBRES_PASAJEROS = {
    2: "Jorge Abraham",
    3: "Alexa Gutierrez",
    4: "Blanca Gutierrez",
    5: "Abril Diaz",
}

# Lista de respaldo para registros antiguos que no tengan un id_usuario o nombre asignado
NOMBRES_RESPALDO = [
    "Jorge Abraham",
    "Abril Diaz",
    "Alexa Gutierrez",
    "Blanca Gutierrez",
]


def _es_admin(solicitud: Request) -> bool:
    """Detecta si la petición pertenece a una sesión de Administrador."""
    rol_query = (solicitud.query_params.get("rol") or "").strip().lower()
    rol_header = (solicitud.headers.get("x-velox-rol") or "").strip().lower()
    rol_cookie = (solicitud.cookies.get("velox_usuario_rol") or "").strip().lower()
    return "admin" in rol_query or "admin" in rol_header or "admin" in rol_cookie


def _resolver_nombre_pasajero(doc: dict, indice: int) -> str:
    """Función auxiliar que obtiene el nombre real del pasajero en lugar de mostrar 'Usuario ID'."""
    nombre_guardado = str(doc.get("nombre_pasajero") or doc.get("pasajero") or "").strip()
    if nombre_guardado and not nombre_guardado.lower().startswith("usuario"):
        return nombre_guardado

    id_u = doc.get("id_usuario")
    if id_u is not None and str(id_u).strip() != "":
        try:
            id_int = int(id_u)
            usuario = db["usuario"].find_one({"id_usuario": id_int}, {"_id": 0})
            if usuario:
                nombre = str(usuario.get("nombre", "")).strip()
                apellido = str(usuario.get("apellido_paterno", "")).strip()
                completo = f"{nombre} {apellido}".strip()
                if completo and not completo.lower().startswith("usuario"):
                    return completo
            if id_int in MAPA_NOMBRES_PASAJEROS:
                return MAPA_NOMBRES_PASAJEROS[id_int]
        except (TypeError, ValueError):
            pass

    return NOMBRES_RESPALDO[indice % len(NOMBRES_RESPALDO)]


def _extraer_estrellas(doc: dict) -> int:
    """Función auxiliar para normalizar la puntuación recibida en un entero válido de 1 a 5 estrellas."""
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
    """Función auxiliar para ordenar de forma segura las reseñas de la más reciente a la más antigua."""
    try:
        return int(doc.get("id_calificacion", 0))
    except (TypeError, ValueError):
        return 0


@router.get("/")
def listar_calificaciones(solicitud: Request):
    """Ruta HTTP GET para mostrar la vista principal de calificaciones."""
    lista_raw = CalificacionController.obtener_todas_las_calificaciones()
    es_admin = _es_admin(solicitud)

    if es_admin:
        calificaciones_admin = [
            item.model_dump() if hasattr(item, "model_dump") else (item.dict() if hasattr(item, "dict") else dict(item))
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

    lista_calificaciones = []
    suma_estrellas = 0

    for idx, item in enumerate(lista_raw):
        doc = item.model_dump() if hasattr(item, "model_dump") else (item.dict() if hasattr(item, "dict") else dict(item))

        if doc.get("tipo") == "conductor_a_pasajero":
            continue

        estrellas = _extraer_estrellas(doc)
        suma_estrellas += estrellas
        nombre_pasajero = _resolver_nombre_pasajero(doc, idx)
        id_cal = doc.get("id_calificacion", idx + 1)

        if estrellas >= 4:
            etiqueta, clase_badge, icono = "Excelente", "bg-success-subtle text-success", "bi-emoji-smile-fill"
        elif estrellas == 3:
            etiqueta, clase_badge, icono = "Regular", "bg-warning-subtle text-dark", "bi-emoji-neutral-fill"
        else:
            etiqueta, clase_badge, icono = "A mejorar", "bg-danger-subtle text-danger", "bi-emoji-frown-fill"

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
    promedio = round(suma_estrellas / len(lista_calificaciones), 1) if lista_calificaciones else 4.8

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
    nueva_calificacion = CalificacionModel(
        puntuacion=puntuacion,
        comentario=comentario,
        id_usuario=id_usuario,
        id_viaje=id_viaje,
    )

    CalificacionController.crear_calificacion(nueva_calificacion)

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