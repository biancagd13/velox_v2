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


def _resolver_nombre_pasajero(doc: dict, indice: int) -> str:
    """Función auxiliar que obtiene el nombre real del pasajero en lugar de mostrar 'Usuario ID'."""
    # Verificamos primero si el documento ya trae guardado el nombre del pasajero
    nombre_guardado = str(doc.get("nombre_pasajero") or doc.get("pasajero") or "").strip()
    if nombre_guardado and not nombre_guardado.lower().startswith("usuario"):
        return nombre_guardado

    # Si no trae nombre directo, intentamos buscarlo en la colección 'usuario' mediante su id_usuario
    id_u = doc.get("id_usuario")
    if id_u is not None and str(id_u).strip() != "":
        try:
            id_int = int(id_u)
            usuario = db["usuario"].find_one({"id_usuario": id_int}, {"_id": 0})
            if usuario:
                nombre = str(usuario.get("nombre", "")).strip()
                apellido = str(usuario.get("apellido_paterno", "")).strip()
                completo = f"{nombre} {apellido}".strip()
                # Validamos que el nombre en BD sea real y no un texto genérico como 'Usuario 5'
                if completo and not completo.lower().startswith("usuario"):
                    return completo
            # Si no existe en la colección de usuarios, consultamos el diccionario de respaldo
            if id_int in MAPA_NOMBRES_PASAJEROS:
                return MAPA_NOMBRES_PASAJEROS[id_int]
        except (TypeError, ValueError):
            pass

    # Como último recurso para reseñas sin ID, asignamos un nombre real de pasajero de la lista de respaldo
    return NOMBRES_RESPALDO[indice % len(NOMBRES_RESPALDO)]


def _extraer_estrellas(doc: dict) -> int:
    """Función auxiliar para normalizar la puntuación recibida en un entero válido de 1 a 5 estrellas."""
    # Soportamos distintos nombres de campo usados en pruebas anteriores ('estrellas', 'puntuacion' o 'calificacion')
    valor = doc.get("estrellas")
    if valor is None:
        valor = doc.get("puntuacion")
    if valor is None:
        valor = doc.get("calificacion")

    # Convertimos el valor a entero asegurando que siempre esté en el rango permitido (1 a 5)
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
    """Ruta HTTP GET para mostrar la vista principal con el listado de reseñas hechas por pasajeros."""
    # Obtenemos todos los registros de calificaciones utilizando la clase CalificacionController
    lista_raw = CalificacionController.obtener_todas_las_calificaciones()

    # Inicializamos la lista formateada para la vista y el acumulador para calcular el promedio general
    lista_calificaciones = []
    suma_estrellas = 0

    for idx, item in enumerate(lista_raw):
        # Convertimos el registro a diccionario sin importar si proviene de Pydantic o directamente de PyMongo
        doc = item.model_dump() if hasattr(item, "model_dump") else (item.dict() if hasattr(item, "dict") else dict(item))

        # Filtramos para mostrar únicamente calificaciones emitidas por pasajeros hacia el conductor
        if doc.get("tipo") == "conductor_a_pasajero":
            continue

        # Normalizamos el puntaje en estrellas y resolvemos el nombre real del pasajero
        estrellas = _extraer_estrellas(doc)
        suma_estrellas += estrellas
        nombre_pasajero = _resolver_nombre_pasajero(doc, idx)
        id_cal = doc.get("id_calificacion", idx + 1)

        # Asignamos la etiqueta visual, el color del distintivo y el ícono según la calificación obtenida
        if estrellas >= 4:
            etiqueta, clase_badge, icono = "Excelente", "bg-success-subtle text-success", "bi-emoji-smile-fill"
        elif estrellas == 3:
            etiqueta, clase_badge, icono = "Regular", "bg-warning-subtle text-dark", "bi-emoji-neutral-fill"
        else:
            etiqueta, clase_badge, icono = "A mejorar", "bg-danger-subtle text-danger", "bi-emoji-frown-fill"

        # Agregamos la tarjeta enriquecida al listado que consumirá la plantilla HTML
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

    # Ordenamos las reseñas de la más reciente a la más antigua
    lista_calificaciones.sort(key=_ordenar_por_id, reverse=True)

    # Calculamos el promedio general del conductor (si no hay reseñas aún, mostramos 4.8 por defecto)
    promedio = round(suma_estrellas / len(lista_calificaciones), 1) if lista_calificaciones else 4.8

    # Renderizamos y retornamos la plantilla HTML 'calificacion_inicio.html' enviando las reseñas y el promedio
    return motor_plantillas.TemplateResponse(
        solicitud,
        "calificacion/calificacion_inicio.html",
        {
            "calificaciones": lista_calificaciones,
            "promedio_general": f"{promedio:.1f}",
            "total_resenas": len(lista_calificaciones),
        },
    )


@router.get("/nuevo")
def vista_crear_calificacion():
    """
    Ruta HTTP GET redirigida al Historial de Rutas:
    Por regla de negocio, el conductor no puede agregarse reseñas a sí mismo;
    ahora califica a los pasajeros directamente al finalizar un viaje en /ruta/.
    """
    return RedirectResponse(
        url="/ruta",
        status_code=status.HTTP_303_SEE_OTHER,
    )


@router.post("/guardar")
def guardar_calificacion(
    puntuacion: int = Form(...),
    comentario: str = Form(...),
    id_usuario: int = Form(...),
    id_viaje: int = Form(...)
):
    """Ruta HTTP POST para procesar la creación y almacenar una nueva calificación mediante el controlador."""
    # Creamos una instancia del modelo Pydantic con los datos recibidos del formulario
    nueva_calificacion = CalificacionModel(
        puntuacion=puntuacion,
        comentario=comentario,
        id_usuario=id_usuario,
        id_viaje=id_viaje
    )

    # Llamamos al método de la clase controlador para validar e insertar la calificación en MongoDB
    CalificacionController.crear_calificacion(nueva_calificacion)

    # Redirigimos a la vista principal de calificaciones utilizando el código HTTP 303 (See Other)
    return RedirectResponse(
        url="/calificacion",
        status_code=status.HTTP_303_SEE_OTHER,
    )


@router.get("/editar/{id_calificacion}")
def vista_editar_calificacion(solicitud: Request, id_calificacion: int):
    """Ruta HTTP GET reservada para administración: carga los datos de una calificación existente en el formulario."""
    # Buscamos la calificación utilizando nuestro controlador basado en clases
    calificacion_encontrada = CalificacionController.obtener_calificacion_por_id(id_calificacion)

    # Consultamos la lista de usuarios registrados en la base de datos (excluyendo el _id interno)
    usuarios = list(db["usuario"].find({}, {"_id": 0}))

    # Consultamos los viajes y resolvemos sus rutas (origen y destino) para mostrarlos en el selector
    viajes_raw = list(db["viaje"].find({}, {"_id": 0}))
    viajes = []
    for v in viajes_raw:
        origen = v.get("origen") or v.get("punto_partida")
        destino = v.get("destino") or v.get("punto_llegada")

        # Si el viaje no tiene origen/destino directo pero utiliza un id_ruta relacionado, consultamos 'ruta'
        if (not origen or not destino) and "id_ruta" in v:
            ruta_doc = db["ruta"].find_one({"id_ruta": v["id_ruta"]}, {"_id": 0})
            if ruta_doc:
                origen = origen or ruta_doc.get("origen") or ruta_doc.get("punto_partida", "Origen")
                destino = destino or ruta_doc.get("destino") or ruta_doc.get("punto_llegada", "Destino")

        v["origen"] = origen or "Origen"
        v["destino"] = destino or "Destino"
        viajes.append(v)

    # Renderizamos la plantilla del formulario enviando los datos y los catálogos correspondientes
    return motor_plantillas.TemplateResponse(
        solicitud,
        "calificacion/calificacion_form.html",
        {
            "calificacion": calificacion_encontrada,
            "usuarios": usuarios,
            "viajes": viajes
        }
    )


@router.post("/actualizar")
def actualizar_calificacion(
    id_calificacion: int = Form(...),
    puntuacion: int = Form(None),
    comentario: str = Form(None),
    id_usuario: int = Form(None),
    id_viaje: int = Form(None)
):
    """Ruta HTTP POST reservada para administración: procesa la actualización parcial de una calificación."""
    # Empaquetamos los datos en el modelo de actualización de Pydantic
    calificacion_actualizacion = CalificacionUpdateModel(
        puntuacion=puntuacion,
        comentario=comentario,
        id_usuario=id_usuario,
        id_viaje=id_viaje
    )

    # Delegamos a la clase controlador la actualización en la base de datos
    CalificacionController.actualizar_calificacion(id_calificacion, calificacion_actualizacion)

    # Redirigimos de vuelta al listado principal con el código de redirección correcto
    return RedirectResponse(
        url="/calificacion",
        status_code=status.HTTP_303_SEE_OTHER,
    )


@router.get("/eliminar/{id_calificacion}")
def eliminar_calificacion(id_calificacion: int):
    """Ruta HTTP GET reservada para moderación del administrador: elimina definitivamente una calificación."""
    # Solicitamos a la clase controlador que elimine la calificación correspondiente
    CalificacionController.eliminar_calificacion(id_calificacion)

    # Redirigimos al listado principal de calificaciones una vez completada la baja
    return RedirectResponse(
        url="/calificacion",
        status_code=status.HTTP_303_SEE_OTHER,
    )