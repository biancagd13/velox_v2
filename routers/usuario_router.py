# Importación de la clase controladora completa para gestionar toda la lógica de negocio y operaciones de la entidad usuario
from controllers.usuario_controller import UsuarioController

# Importación de los modelos de validación de Pydantic necesarios para asegurar la integridad de los datos
from models.usuario_model import UsuarioModel, UsuarioUpdateModel

# Importación de la instancia de la base de datos para realizar consultas directas
from database import db

# Importación de las herramientas esenciales de FastAPI: APIRouter para subrutas, Body/Form para recepción de datos y Request para solicitudes HTTP
from fastapi import APIRouter, Body, Form, Request, status

# Importación de respuestas HTTP y Jinja2Templates para renderizar plantillas HTML
from fastapi.responses import JSONResponse, RedirectResponse
from fastapi.templating import Jinja2Templates

# Inicialización y configuración del enrutador con un prefijo común y una etiqueta descriptiva en español para la documentación OpenAPI
router = APIRouter(prefix="/usuario", tags=["Usuarios"])

# Configuración del directorio base donde se encuentran almacenadas las plantillas HTML (carpeta 'templates')
motor_plantillas = Jinja2Templates(directory="templates")


def _to_int(valor, default: int = 0) -> int:
    try:
        return int(valor)
    except (TypeError, ValueError):
        return default


def _to_float(valor, default: float = 0.0) -> float:
    try:
        return float(valor)
    except (TypeError, ValueError):
        return default


def _obtener_id_usuario_activo(solicitud: Request | None = None) -> int:
    """Obtiene el id_usuario desde query, header o cookie; si no viene, usa el conductor activo (5)."""
    if solicitud is not None:
        raw = (
            solicitud.query_params.get("id_usuario")
            or solicitud.headers.get("x-velox-usuario-id")
            or solicitud.cookies.get("velox_usuario_id")
            or ""
        ).strip()
        if raw.isdigit() and int(raw) > 0:
            return int(raw)
    return 5


def _calcular_rendimiento_real(id_usuario: int) -> dict:
    """Calcula racha/promedio, viajes y ganancias reales del usuario/conductor desde MongoDB."""
    if not id_usuario or id_usuario <= 0:
        return {
            "calificacion": "0.0",
            "viajes_completados": 0,
            "ganancias_estimadas": 0,
            "viajes": [],
            "ids_viajes": [],
        }

    filtro_dueno = {
        "$or": [
            {"id_usuario": id_usuario},
            {"id_conductor": id_usuario},
            {"usuario_id": id_usuario},
            {"conductor_id": id_usuario},
        ]
    }

    rutas = list(db["ruta"].find(filtro_dueno, {"_id": 0, "id_ruta": 1}))
    ids_rutas = [_to_int(r.get("id_ruta")) for r in rutas if r.get("id_ruta") is not None]
    ids_rutas = [x for x in ids_rutas if x > 0]

    condiciones_viaje = [
        {"id_usuario": id_usuario},
        {"id_conductor": id_usuario},
        {"usuario_id": id_usuario},
        {"conductor_id": id_usuario},
    ]
    if ids_rutas:
        condiciones_viaje.append({"id_ruta": {"$in": ids_rutas}})

    viajes = list(db["viaje"].find({"$or": condiciones_viaje}, {"_id": 0}))
    ids_viajes = [_to_int(v.get("id_viaje")) for v in viajes if v.get("id_viaje") is not None]
    ids_viajes = [x for x in ids_viajes if x > 0]

    total_viajes = len(viajes)
    total_ganancias = 0.0

    for v in viajes:
        monto = (
            v.get("costo")
            if v.get("costo") is not None
            else (
                v.get("aportacion")
                if v.get("aportacion") is not None
                else (
                    v.get("precio")
                    if v.get("precio") is not None
                    else v.get("tarifa")
                )
            )
        )
        total_ganancias += _to_float(monto, default=0.0)

    condiciones_calif = [
        {"id_conductor": id_usuario},
        {"id_conductor_viaje": id_usuario},
        {"conductor_id": id_usuario},
    ]
    if ids_viajes:
        condiciones_calif.append({"id_viaje": {"$in": ids_viajes}})
    if ids_rutas:
        condiciones_calif.append({"id_ruta": {"$in": ids_rutas}})

    calificaciones = list(
        db["calificacion"].find(
            {"$or": condiciones_calif, "tipo": {"$ne": "conductor_a_pasajero"}},
            {"_id": 0, "puntuacion": 1, "estrellas": 1, "calificacion": 1},
        )
    )

    puntos = []
    for c in calificaciones:
        raw_val = (
            c.get("puntuacion")
            if c.get("puntuacion") is not None
            else (
                c.get("estrellas")
                if c.get("estrellas") is not None
                else c.get("calificacion")
            )
        )
        val = _to_float(raw_val, default=-1.0)
        if val >= 0:
            puntos.append(val)

    promedio = f"{(sum(puntos) / len(puntos)):.1f}" if puntos else "0.0"
    ganancias_limpias = (
        int(total_ganancias)
        if total_ganancias.is_integer()
        else round(total_ganancias, 2)
    )

    return {
        "calificacion": promedio,
        "viajes_completados": total_viajes,
        "ganancias_estimadas": ganancias_limpias,
        "viajes": viajes,
        "ids_viajes": ids_viajes,
    }


def _construir_usuario_enriquecido(id_usuario: int) -> dict:
    """Obtiene el documento real del usuario en MongoDB y le adjunta su rendimiento real."""
    doc = db["usuario"].find_one({"id_usuario": id_usuario}, {"_id": 0})
    if not doc:
        doc = {
            "id_usuario": id_usuario,
            "nombre": "",
            "apellido_paterno": "",
            "apellido_materno": "",
            "rol": "Conductor",
            "matricula": "",
            "correo_institucional": "",
            "telefono": "",
            "contrasena": "",
        }

    rendimiento = _calcular_rendimiento_real(id_usuario)

    return {
        **doc,
        "calificacion": rendimiento["calificacion"],
        "viajes_completados": rendimiento["viajes_completados"],
        "ganancias_estimadas": rendimiento["ganancias_estimadas"],
    }


@router.post("/api/login")
def login_dinamico(payload: dict = Body(...)):
    """
    Autentica al usuario contra MongoDB y devuelve su nombre, rol e ID
    detectados automáticamente desde las colecciones 'usuario' y 'rol'.
    """
    correo = (payload.get("correo") or "").strip()
    password = payload.get("password") or ""

    if not correo or not password:
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={"ok": False, "mensaje": "Ingresa tu correo institucional y contraseña."},
        )

    usuario_autenticado = UsuarioController.autenticar_usuario(correo, password)

    if not usuario_autenticado:
        return JSONResponse(
            status_code=status.HTTP_401_UNAUTHORIZED,
            content={"ok": False, "mensaje": "Correo institucional o contraseña incorrectos."},
        )

    rol_texto = (usuario_autenticado.get("rol") or "").strip().lower()
    if "admin" in rol_texto:
        rol_normalizado = "Administrador"
        destino = "/perfil"
    elif "conductor" in rol_texto:
        rol_normalizado = "Conductor"
        destino = "/perfil-conductor"
    else:
        rol_normalizado = "Pasajero"
        destino = "/pasajero/"

    usuario_autenticado["rol"] = rol_normalizado

    return {
        "ok": True,
        "usuario": usuario_autenticado,
        "destino": destino,
    }


@router.get("/api/{id_usuario}")
def obtener_perfil_api(id_usuario: int):
    """Devuelve los datos reales del usuario y su rendimiento calculado desde MongoDB."""
    usuario_data = _construir_usuario_enriquecido(id_usuario)
    return {"ok": True, "usuario": usuario_data}


@router.put("/api/{id_usuario}/contacto")
def actualizar_contacto_api(id_usuario: int, payload: dict = Body(...)):
    """Actualiza únicamente el teléfono y el correo institucional del usuario en MongoDB."""
    telefono = (payload.get("telefono") or "").strip()
    correo = (payload.get("correo_institucional") or payload.get("correo") or "").strip()

    if not telefono or not correo:
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={"ok": False, "mensaje": "El teléfono y el correo son obligatorios."},
        )

    db["usuario"].update_one(
        {"id_usuario": id_usuario},
        {"$set": {"telefono": telefono, "correo_institucional": correo}},
    )

    return {
        "ok": True,
        "usuario": {
            "id_usuario": id_usuario,
            "telefono": telefono,
            "correo_institucional": correo,
        },
    }


@router.put("/api/{id_usuario}/password")
def actualizar_password_api(id_usuario: int, payload: dict = Body(...)):
    """Actualiza la contraseña del usuario en MongoDB."""
    nueva_password = (payload.get("contrasena") or payload.get("password") or "").strip()
    if len(nueva_password) < 4:
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={"ok": False, "mensaje": "La contraseña debe tener al menos 4 caracteres."},
        )

    db["usuario"].update_one(
        {"id_usuario": id_usuario},
        {"$set": {"contrasena": nueva_password}},
    )

    return {"ok": True, "mensaje": "Contraseña actualizada."}


@router.get("/")
def listar_usuarios(solicitud: Request):
    """
    Ruta HTTP GET para mostrar la vista principal de gestión,
    recuperando el listado completo de todos los usuarios registrados en el sistema.
    """
    lista_usuarios = UsuarioController.obtener_todos_los_usuarios()

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
    roles = list(db["rol"].find({}, {"_id": 0}))

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
    try:
        id_rol_int = int(id_rol)
    except (ValueError, TypeError):
        id_rol_int = 1

    nuevo_usuario = UsuarioModel(
        matricula=matricula,
        nombre=nombre,
        apellido_paterno=apellido_paterno,
        apellido_materno=apellido_materno,
        correo_institucional=correo_institucional,
        telefono=telefono,
        contrasena=contrasena,
        id_rol=id_rol_int,
        fecha_registro=None
    )

    UsuarioController.crear_usuario(nuevo_usuario)

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
    usuario_encontrado = UsuarioController.obtener_usuario_por_id(id_usuario)
    roles = list(db["rol"].find({}, {"_id": 0}))

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

    UsuarioController.actualizar_usuario(id_usuario, usuario_actualizacion)

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
    UsuarioController.eliminar_usuario(id_usuario)

    return RedirectResponse(
        url="/usuario",
        status_code=status.HTTP_303_SEE_OTHER,
    )


# =========================================================================
# RUTA: PERFIL DEL CONDUCTOR E HISTORIAL DE CALIFICACIONES
# =========================================================================
@router.get("/perfil-conductor")
def vista_perfil_conductor(solicitud: Request):
    """
    Ruta HTTP GET para mostrar el perfil dinámico del conductor con datos reales
    de usuario, viajes, ganancias y calificaciones desde MongoDB.
    """
    id_conductor = _obtener_id_usuario_activo(solicitud)
    conductor = _construir_usuario_enriquecido(id_conductor)

    rendimiento = _calcular_rendimiento_real(id_conductor)
    ids_viajes = rendimiento["ids_viajes"]

    historial_calificaciones = []

    if ids_viajes:
        calificaciones_raw = list(
            db["calificacion"]
            .find({"id_viaje": {"$in": ids_viajes}}, {"_id": 0})
            .sort("id_calificacion", -1)
            .limit(10)
        )

        for calif in calificaciones_raw:
            id_pasajero = calif.get("id_usuario")
            pasajero = db["usuario"].find_one({"id_usuario": id_pasajero}, {"_id": 0})

            if pasajero:
                nombre = pasajero.get("nombre", "")
                apellido = pasajero.get("apellido_paterno", "")
                calif["nombre_pasajero"] = f"{nombre} {apellido}".strip()
                calif["matricula_pasajero"] = pasajero.get("matricula", "Sin matrícula")
            else:
                calif["nombre_pasajero"] = "Usuario Anónimo"
                calif["matricula_pasajero"] = "Sin registro"

            calif["puntuacion"] = calif.get("puntuacion", 5)
            historial_calificaciones.append(calif)

    return motor_plantillas.TemplateResponse(
        solicitud,
        "perfiles/perfil_conductor.html",
        {
            "usuario": conductor,
            "historial_calificaciones": historial_calificaciones,
        },
    )