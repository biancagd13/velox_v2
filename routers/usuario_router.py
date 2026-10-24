# Importación de la clase controladora completa para gestionar toda la lógica de negocio y operaciones de la entidad usuario
from controllers.usuario_controller import UsuarioController

# Importación de los modelos de validación de Pydantic necesarios para asegurar la integridad de los datos
from models.usuario_model import UsuarioModel, UsuarioUpdateModel

# Importación de la instancia de la base de datos para realizar consultas directas (como la obtención de roles)
from database import db

# Importación de las herramientas esenciales de FastAPI: APIRouter para subrutas, Form para recepción de formularios y Request para solicitudes HTTP
from fastapi import APIRouter, Form, Request, status

# Importación de RedirectResponse para gestionar redirecciones seguras entre endpoints y Jinja2Templates para renderizar plantillas HTML
from fastapi.responses import RedirectResponse
from fastapi.templating import Jinja2Templates

# Inicialización y configuración del enrutador con un prefijo común y una etiqueta descriptiva en español para la documentación OpenAPI
router = APIRouter(prefix="/usuario", tags=["Usuarios"])

# Configuración del directorio base donde se encuentran almacenadas las plantillas HTML (carpeta 'templates')
motor_plantillas = Jinja2Templates(directory="templates")


@router.get("/")
def listar_usuarios(solicitud: Request):
    """
    Ruta HTTP GET para mostrar la vista principal de gestión, 
    recuperando el listado completo de todos los usuarios registrados en el sistema.
    """
    # Solicitamos al controlador la lista completa de registros enriquecidos con sus roles
    lista_usuarios = UsuarioController.obtener_todos_los_usuarios()
    
    # Renderizamos y retornamos la plantilla de inicio pasando la solicitud y la lista de registros
    return motor_plantillas.TemplateResponse(
        solicitud,
        "usuario/usuario_inicio.html",
        {
            "usuarios": lista_usuarios,
        },
    )


@router.get("/nuevo")
def vista_crear_usuario(solicitud: Request):
    """
    Ruta HTTP GET para cargar la vista del formulario vacío 
    destinado al registro e inserción de un nuevo usuario en el sistema.
    """
    # Consultamos todos los roles disponibles en la colección de la base de datos (excluyendo el _id interno)
    roles = list(db["rol"].find({}, {"_id": 0}))
    
    # Renderizamos el formulario enviando 'usuario' como None (indicando alta) y la lista de roles para el select
    return motor_plantillas.TemplateResponse(
        solicitud, "usuario/usuario_form.html", {"usuario": None, "roles": roles}
    )


@router.post("/guardar")
def guardar_usuario(
    matricula: str = Form(...),
    nombre: str = Form(...),
    apellido_paterno: str = Form(...),
    apellido_materno: str = Form(None),
    correo_institucional: str = Form(...),
    telefono: str = Form(None),
    contrasena: str = Form(...),
    id_rol: str = Form(...)
):
    """
    Ruta HTTP POST para procesar los datos enviados desde el formulario de creación, 
    validarlos mediante Pydantic y almacenar el nuevo registro de manera segura en MongoDB.
    """
    
    # Conversión segura del id_rol recibido por el menú desplegable (select) a un valor de tipo entero
    try:
        id_rol_int = int(id_rol)
    except (ValueError, TypeError):
        id_rol_int = 1  # Asignación de un rol por defecto en caso de cualquier imprevisto

    # Creación y validación de una instancia formal utilizando el modelo Pydantic del sistema
    nuevo_usuario = UsuarioModel(
        matricula=matricula,
        nombre=nombre,
        apellido_paterno=apellido_paterno,
        apellido_materno=apellido_materno,
        correo_institucional=correo_institucional,
        telefono=telefono,
        contrasena=contrasena,
        id_rol=id_rol_int,
        fecha_registro=None  # El controlador se encarga de autocompletar la fecha y hora exacta si se omite
    )

    # Delegamos al controlador la inserción física del registro validado en la base de datos
    UsuarioController.crear_usuario(nuevo_usuario)

    # Redirección automática de vuelta al listado principal utilizando el código HTTP estándar 303 (See Other)
    return RedirectResponse(
        url="/usuario",
        status_code=status.HTTP_303_SEE_OTHER,
    )


@router.get("/editar/{id_usuario}")
def vista_editar_usuario(solicitud: Request, id_usuario: int):
    """
    Ruta HTTP GET para buscar y cargar los datos de un usuario existente 
    dentro del formulario para realizar modificaciones o actualizaciones.
    """
    # Buscamos el registro específico en la base de datos a través de su identificador numérico único
    usuario_encontrado = UsuarioController.obtener_usuario_por_id(id_usuario)
    
    # Consultamos nuevamente la lista de roles para mantener activo el menú desplegable en modo edición
    roles = list(db["rol"].find({}, {"_id": 0}))

    # Renderizamos la plantilla del formulario enviando los datos del usuario encontrado y los roles disponibles
    return motor_plantillas.TemplateResponse(
        solicitud, "usuario/usuario_form.html", {"usuario": usuario_encontrado, "roles": roles}
    )


@router.post("/actualizar")
def actualizar_usuario(
    id_usuario: int = Form(...),
    matricula: str = Form(None),
    nombre: str = Form(None),
    apellido_paterno: str = Form(None),
    apellido_materno: str = Form(None),
    correo_institucional: str = Form(None),
    telefono: str = Form(None),
    contrasena: str = Form(None),
    id_rol: int = Form(None)
):
    """
    Ruta HTTP POST para procesar y aplicar la actualización parcial de un usuario existente 
    utilizando el modelo unificado de actualización.
    """
    # Empaquetamos los campos recibidos dentro del modelo Pydantic de actualización parcial
    usuario_actualizacion = UsuarioUpdateModel(
        matricula=matricula,
        nombre=nombre,
        apellido_paterno=apellido_paterno,
        apellido_materno=apellido_materno,
        correo_institucional=correo_institucional,
        telefono=telefono,
        contrasena=contrasena,
        id_rol=id_rol
    )

    # Delegamos al controlador la ejecución de la actualización en la base de datos
    UsuarioController.actualizar_usuario(id_usuario, usuario_actualizacion)

    # Redirección formal al listado principal una vez aplicada la modificación
    return RedirectResponse(
        url="/usuario",
        status_code=status.HTTP_303_SEE_OTHER,
    )


@router.get("/eliminar/{id_usuario}")
def eliminar_usuario(id_usuario: int):
    """
    Ruta HTTP GET para procesar la baja definitiva de un registro de usuario 
    y ejecutar su respectiva eliminación en cascada en las colecciones vinculadas.
    """
    # Solicitamos al controlador la eliminación permanente del registro por su ID
    UsuarioController.eliminar_usuario(id_usuario)

    # Redirección final al listado general de usuarios
    return RedirectResponse(
        url="/usuario",
        status_code=status.HTTP_303_SEE_OTHER,
    )