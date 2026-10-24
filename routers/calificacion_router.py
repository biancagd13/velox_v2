# Importamos la clase controladora completa para gestionar la lógica de negocio de las calificaciones
from controllers.calificacion_controller import CalificacionController

# Importamos los modelos de Pydantic necesarios para validar los datos de la calificación
from models.calificacion_model import CalificacionModel, CalificacionUpdateModel

# Importamos la base de datos para poder consultar los usuarios y viajes registrados
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


@router.get("/")
def listar_calificaciones(solicitud: Request):
    """Ruta HTTP GET para mostrar la vista principal con el listado de todas las calificaciones."""
    # Obtenemos todos los registros de calificaciones utilizando la clase CalificacionController
    lista_calificaciones = CalificacionController.obtener_todas_las_calificaciones()

    # Renderizamos y retornamos la plantilla HTML 'calificacion_inicio.html' enviando las calificaciones obtenidas
    return motor_plantillas.TemplateResponse(
        solicitud,
        "calificacion/calificacion_inicio.html",
        {
            "calificaciones": lista_calificaciones,
        },
    )


@router.get("/nuevo")
def vista_crear_calificacion(solicitud: Request):
    """Ruta HTTP GET para cargar el formulario vacío destinado a registrar una nueva calificación."""
    # Consultamos los usuarios registrados en la base de datos (excluyendo el _id interno)
    usuarios = list(db["usuario"].find({}, {"_id": 0}))
    
    # Consultamos los viajes y resolvemos sus rutas de forma robusta
    viajes_raw = list(db["viaje"].find({}, {"_id": 0}))
    viajes = []
    for v in viajes_raw:
        origen = v.get("origen") or v.get("punto_partida")
        destino = v.get("destino") or v.get("punto_llegada")
        
        # Si el viaje no tiene origen/destino directo pero utiliza un id_ruta relacionado
        if (not origen or not destino) and "id_ruta" in v:
            ruta_doc = db["ruta"].find_one({"id_ruta": v["id_ruta"]}, {"_id": 0})
            if ruta_doc:
                origen = origen or ruta_doc.get("origen") or ruta_doc.get("punto_partida", "Origen")
                destino = destino or ruta_doc.get("destino") or ruta_doc.get("punto_llegada", "Destino")
        
        v["origen"] = origen or "Origen"
        v["destino"] = destino or "Destino"
        viajes.append(v)

    # Renderizamos la plantilla de formulario pasando los catálogos ya enriquecidos para los selectores
    return motor_plantillas.TemplateResponse(
        solicitud, 
        "calificacion/calificacion_form.html", 
        {
            "calificacion": None, 
            "usuarios": usuarios, 
            "viajes": viajes
        }
    )


@router.post("/guardar")
def guardar_calificacion(
    puntuacion: int = Form(...),
    comentario: str = Form(...),
    id_usuario: int = Form(...),
    id_viaje: int = Form(...)
):
    """Ruta HTTP POST para procesar el formulario de creación y almacenar la nueva calificación."""
    # Creamos una instancia del modelo Pydantic con los datos recibidos del formulario
    nueva_calificacion = CalificacionModel(
        puntuacion=puntuacion,
        comentario=comentario,
        id_usuario=id_usuario,
        id_viaje=id_viaje
    )

    # Llamamos al método de la clase controlador para validar e insertar la calificación
    CalificacionController.crear_calificacion(nueva_calificacion)

    # Redirigimos al usuario a la vista principal de calificaciones utilizando el código HTTP 303 (See Other)
    return RedirectResponse(
        url="/calificacion",
        status_code=status.HTTP_303_SEE_OTHER,
    )


@router.get("/editar/{id_calificacion}")
def vista_editar_calificacion(solicitud: Request, id_calificacion: int):
    """Ruta HTTP GET para cargar los datos de una calificación existente dentro del formulario de edición."""
    # Buscamos la calificación utilizando nuestro controlador basado en clases
    calificacion_encontrada = CalificacionController.obtener_calificacion_por_id(id_calificacion)
    
    # Consultamos la lista de usuarios
    usuarios = list(db["usuario"].find({}, {"_id": 0}))
    
    # Realizamos el mismo proceso de resolución de rutas para el modo edición
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
    """Ruta HTTP POST para procesar la actualización parcial de una calificación existente."""
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
    """Ruta HTTP GET para procesar la eliminación definitiva de una calificación."""
    # Solicitamos a la clase controlador que elimine la calificación correspondiente
    CalificacionController.eliminar_calificacion(id_calificacion)

    # Redirigimos al listado principal de calificaciones una vez completada la baja
    return RedirectResponse(
        url="/calificacion",
        status_code=status.HTTP_303_SEE_OTHER,
    )