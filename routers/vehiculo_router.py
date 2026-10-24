# Importamos la clase controladora completa para gestionar la lógica de negocio de los vehículos
from controllers.vehiculo_controller import VehiculoController

# Importamos los modelos de Pydantic necesarios para validar los datos del vehículo
from models.vehiculo_model import VehiculoModel, VehiculoUpdateModel

# Importamos la base de datos para poder consultar los usuarios registrados
from database import db

# Importamos las herramientas de FastAPI: APIRouter para rutas, Form para datos de formularios y Request para solicitudes HTTP
from fastapi import APIRouter, Form, Request, status

# Importamos RedirectResponse para redirigir entre endpoints y Jinja2Templates para renderizar plantillas HTML
from fastapi.responses import RedirectResponse
from fastapi.templating import Jinja2Templates

# Inicializamos el enrutador con el prefijo y etiqueta correspondientes en español
router = APIRouter(prefix="/vehiculo", tags=["Vehículo"])

# Configuramos el directorio base donde se encuentran las plantillas HTML (carpeta 'templates')
motor_plantillas = Jinja2Templates(directory="templates")


@router.get("/")
def listar_vehiculos(solicitud: Request):
    """Ruta HTTP GET para mostrar la vista principal con el listado de todos los vehículos."""
    # Obtenemos todos los registros de vehículos utilizando la clase VehiculoController
    lista_vehiculos = VehiculoController.obtener_todos_los_vehiculos()

    # Renderizamos y retornamos la plantilla HTML 'vehiculo_inicio.html' enviando los vehículos obtenidos
    return motor_plantillas.TemplateResponse(
        solicitud,
        "vehiculo/vehiculo_inicio.html",
        {
            "vehiculos": lista_vehiculos,
        },
    )


@router.get("/nuevo")
def vista_crear_vehiculo(solicitud: Request):
    """Ruta HTTP GET para cargar el formulario vacío destinado a registrar un nuevo vehículo."""
    # Consultamos los usuarios registrados en la base de datos para el selector del propietario
    usuarios = list(db["usuario"].find({}, {"_id": 0}))

    # Renderizamos la plantilla de formulario pasando 'vehiculo' como None y la lista de usuarios
    return motor_plantillas.TemplateResponse(
        solicitud, 
        "vehiculo/vehiculo_form.html", 
        {
            "vehiculo": None, 
            "usuarios": usuarios
        }
    )


@router.post("/guardar")
def guardar_vehiculo(
    placa: str = Form(...),
    nombre: str = Form(...),
    modelo: str = Form(...),
    color: str = Form(...),
    capacidad_asientos: int = Form(...),
    id_usuario: int = Form(...)
):
    """Ruta HTTP POST para procesar el formulario de creación y almacenar el nuevo vehículo."""
    # Creamos una instancia del modelo Pydantic con los datos recibidos del formulario
    nuevo_vehiculo = VehiculoModel(
        placa=placa,
        nombre=nombre,
        modelo=modelo,
        color=color,
        capacidad_asientos=capacidad_asientos,
        id_usuario=id_usuario
    )

    # Llamamos al método de la clase controlador para validar e insertar el vehículo
    VehiculoController.crear_vehiculo(nuevo_vehiculo)

    # Redirigimos al usuario a la vista principal de vehículos utilizando el código HTTP 303 (See Other)
    return RedirectResponse(
        url="/vehiculo",
        status_code=status.HTTP_303_SEE_OTHER,
    )


@router.get("/editar/{id_vehiculo}")
def vista_editar_vehiculo(solicitud: Request, id_vehiculo: int):
    """Ruta HTTP GET para cargar los datos de un vehículo existente dentro del formulario de edición."""
    # Buscamos el vehículo utilizando nuestro controlador basado en clases
    vehiculo_encontrado = VehiculoController.obtener_vehiculo_por_id(id_vehiculo)
    
    # Consultamos la lista de usuarios para mantener activo el selector en modo edición
    usuarios = list(db["usuario"].find({}, {"_id": 0}))

    # Renderizamos la plantilla del formulario enviando los datos del vehículo y los usuarios
    return motor_plantillas.TemplateResponse(
        solicitud, 
        "vehiculo/vehiculo_form.html", 
        {
            "vehiculo": vehiculo_encontrado, 
            "usuarios": usuarios
        }
    )


@router.post("/actualizar")
def actualizar_vehiculo(
    id_vehiculo: int = Form(...),
    placa: str = Form(None),
    nombre: str = Form(None),
    modelo: str = Form(None),
    color: str = Form(None),
    capacidad_asientos: int = Form(None),
    id_usuario: int = Form(None)
):
    """Ruta HTTP POST para procesar la actualización parcial de un vehículo existente."""
    # Empaquetamos los datos en el modelo de actualización de Pydantic
    vehiculo_actualizacion = VehiculoUpdateModel(
        placa=placa,
        nombre=nombre,
        modelo=modelo,
        color=color,
        capacidad_asientos=capacidad_asientos,
        id_usuario=id_usuario
    )

    # Delegamos a la clase controlador la actualización en la base de datos
    VehiculoController.actualizar_vehiculo(id_vehiculo, vehiculo_actualizacion)

    # Redirigimos de vuelta al listado principal con el código de redirección correcto
    return RedirectResponse(
        url="/vehiculo",
        status_code=status.HTTP_303_SEE_OTHER,
    )


@router.get("/eliminar/{id_vehiculo}")
def eliminar_vehiculo(id_vehiculo: int):
    """Ruta HTTP GET para procesar la eliminación definitiva de un vehículo."""
    # Solicitamos a la clase controlador que elimine el vehículo correspondiente
    VehiculoController.eliminar_vehiculo(id_vehiculo)

    # Redirigimos al listado principal de vehículos una vez completada la baja
    return RedirectResponse(
        url="/vehiculo",
        status_code=status.HTTP_303_SEE_OTHER,
    )