from controllers.ruta_controller import RutaController
from models.ruta_model import RutaModel, RutaUpdateModel
from database import db
from fastapi import APIRouter, Form, Request, status
from fastapi.responses import RedirectResponse
from fastapi.templating import Jinja2Templates

# Inicializamos el enrutador con el prefijo y etiqueta correspondientes en español
router = APIRouter(prefix="/ruta", tags=["Ruta"])
motor_plantillas = Jinja2Templates(directory="templates")


@router.get("/")
def listar_rutas(solicitud: Request):
    """Ruta HTTP GET para mostrar la vista principal con el listado de todas las rutas."""
    lista_rutas = RutaController.obtener_todas_las_rutas()
    return motor_plantillas.TemplateResponse(
        solicitud,
        "ruta/ruta_inicio.html",
        {
            "rutas": lista_rutas,
        },
    )


@router.get("/nuevo")
def vista_crear_ruta(solicitud: Request):
    """Ruta HTTP GET para cargar el formulario vacío destinado a registrar una nueva ruta."""
    usuarios = list(db["usuario"].find({}, {"_id": 0}))
    return motor_plantillas.TemplateResponse(
        solicitud, 
        "ruta/ruta_form.html", 
        {
            "ruta": None,
            "usuarios": usuarios
        }
    )


@router.post("/guardar")
def guardar_ruta(
    origen: str = Form(...),
    destino: str = Form(...),
    hora_salida: str = Form(...),
    id_usuario: int = Form(...)
):
    """Ruta HTTP POST para procesar el formulario de creación y almacenar la nueva ruta."""
    nueva_ruta = RutaModel(
        origen=origen,
        destino=destino,
        hora_salida=hora_salida,
        id_usuario=id_usuario
    )

    RutaController.crear_ruta(nueva_ruta)

    return RedirectResponse(
        url="/ruta",
        status_code=status.HTTP_303_SEE_OTHER,
    )


@router.get("/editar/{id_ruta}")
def vista_editar_ruta(solicitud: Request, id_ruta: int):
    """Ruta HTTP GET para cargar los datos de una ruta existente dentro del formulario de edición."""
    ruta_encontrada = RutaController.obtener_ruta_por_id(id_ruta)
    usuarios = list(db["usuario"].find({}, {"_id": 0}))

    return motor_plantillas.TemplateResponse(
        solicitud, 
        "ruta/ruta_form.html", 
        {
            "ruta": ruta_encontrada,
            "usuarios": usuarios
        }
    )


@router.post("/actualizar")
def actualizar_ruta(
    id_ruta: int = Form(...),
    origen: str = Form(None),
    destino: str = Form(None),
    hora_salida: str = Form(None),
    id_usuario: int = Form(None)
):
    """Ruta HTTP POST para procesar la actualización parcial de una ruta existente."""
    ruta_actualizacion = RutaUpdateModel(
        origen=origen,
        destino=destino,
        hora_salida=hora_salida,
        id_usuario=id_usuario
    )

    RutaController.actualizar_ruta(id_ruta, ruta_actualizacion)

    return RedirectResponse(
        url="/ruta",
        status_code=status.HTTP_303_SEE_OTHER,
    )


@router.get("/eliminar/{id_ruta}")
def eliminar_ruta(id_ruta: int):
    """Ruta HTTP GET para procesar la eliminación definitiva de una ruta."""
    RutaController.eliminar_ruta(id_ruta)

    return RedirectResponse(
        url="/ruta",
        status_code=status.HTTP_303_SEE_OTHER,
    )