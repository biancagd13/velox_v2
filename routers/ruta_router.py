# Importamos math para el cálculo geodésico (Haversine) y datetime para sellos de tiempo
import math
from datetime import datetime

# Importamos las herramientas de FastAPI para definir rutas, formularios y redirecciones
from fastapi import APIRouter, Request, Form, status
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates

# Importamos la instancia de base de datos MongoDB configurada en database.py
from database import db

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

# Catálogo de pasajeros de respaldo para rutas históricas que no guarden el nombre del pasajero
PASAJEROS_DEMO_RUTA = [
    {"id_usuario": 5, "nombre": "Abril Diaz", "matricula": "222610848"},
    {"id_usuario": 4, "nombre": "Blanca Gutierrez", "matricula": "222411050"},
    {"id_usuario": 3, "nombre": "Alexa Gutierrez", "matricula": "222050411"},
    {"id_usuario": 2, "nombre": "Jorge Abraham", "matricula": "222310045"},
]


def _estimar_metricas(origen: str, destino: str) -> tuple[float, int, float]:
    """Calcula distancia vial aproximada (km), duración (min) y tarifa cuando una ruta viene en 0."""
    # Obtenemos coordenadas de origen y destino (o valores por defecto si no coincide el nombre)
    c1 = COORDENADAS_NODOS.get(origen, [19.2848, -99.6552])
    c2 = COORDENADAS_NODOS.get(destino, [19.3451, -99.5082])

    # Aplicamos la fórmula de Haversine para calcular la distancia sobre la esfera terrestre
    lat1, lon1 = math.radians(c1[0]), math.radians(c1[1])
    lat2, lon2 = math.radians(c2[0]), math.radians(c2[1])
    a = math.sin((lat2 - lat1) / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin((lon2 - lon1) / 2) ** 2

    # Multiplicamos por factor vial 1.22 para reflejar carreteras reales del Valle de Toluca
    km = round(max(4.5, 6371.0 * (2 * math.asin(math.sqrt(a))) * 1.22), 1)
    minutos = max(12, int(round(km * 2.1)))
    costo = 20.0 if km <= 12.0 else 25.0
    return km, minutos, costo


def _recalcular_calificacion_pasajero(id_usuario: int, nombre_pasajero: str):
    """Recalcula el promedio real del pasajero en db['usuario'] cuando el conductor lo califica."""
    # Construimos filtro flexible por ID o por nombre para sincronizar cualquier registro previo
    filtros = []
    if id_usuario > 0:
        filtros.append({"id_usuario": id_usuario})
    if nombre_pasajero:
        filtros.append({"nombre_pasajero": {"$regex": f"^{nombre_pasajero}$", "$options": "i"}})

    if not filtros:
        return

    # Consultamos todas las calificaciones que los conductores han otorgado a este pasajero
    resenas = list(db["calificacion_pasajero"].find({"$or": filtros}, {"_id": 0}))
    valores = [int(r["estrellas"]) for r in resenas if r.get("estrellas") is not None]
    if not valores:
        return

    # Calculamos el promedio base de estrellas otorgadas por el conductor
    promedio_conductor = sum(valores) / len(valores)

    # Localizamos el documento del pasajero en la colección 'usuario'
    usuario = db["usuario"].find_one({"id_usuario": id_usuario}) if id_usuario > 0 else None
    if not usuario and nombre_pasajero:
        usuario = db["usuario"].find_one({"nombre": {"$regex": f"^{nombre_pasajero}$", "$options": "i"}})

    # Ajustamos su calificación final considerando también el impacto de sus cancelaciones previas
    if usuario:
        cancelaciones = int(usuario.get("cancelaciones", 0))
        calificacion_final = max(1.0, min(5.0, round(promedio_conductor - (cancelaciones * 0.15), 1)))
        db["usuario"].update_one(
            {"_id": usuario["_id"]},
            {"$set": {"calificacion": calificacion_final}},
        )


@router.get("/", response_class=HTMLResponse)
def listar_rutas(solicitud: Request, calificado: int = 0):
    """
    Ruta HTTP GET para mostrar el Historial de Rutas del conductor.
    Unifica los viajes aceptados/finalizados de pasajeros reales (colección 'reserva')
    y las rutas históricas (colección 'ruta') para poder calificar a cada pasajero al concluir el viaje.
    """
    tarjetas = []

    # 1. Consultamos los viajes aceptados o finalizados en 'reserva' (ej. Jorge Abraham)
    reservas_activas = list(
        db["reserva"].find(
            {"estado": {"$in": ["aceptada", "confirmada", "finalizada", "Aceptada", "Finalizada"]}},
            {"_id": 0},
        )
    )
    reservas_activas.sort(key=lambda r: r.get("id_reserva", 0), reverse=True)

    for res in reservas_activas:
        id_res = int(res.get("id_reserva", 0))
        nombre_p = str(res.get("nombre_pasajero") or "Jorge Abraham").strip()
        id_u = int(res.get("id_usuario", 2))
        origen = res.get("origen") or "Rectoría UTVT"
        destino = res.get("destino") or "Terminal Toluca"

        # Estimamos métricas faltantes para evitar mostrar valores en 0
        km_est, min_est, costo_est = _estimar_metricas(origen, destino)
        distancia = float(res.get("distancia_km") or km_est)
        tiempo = int(res.get("tiempo_min") or min_est)
        costo = float(res.get("costo") or costo_est)

        # Verificamos si el conductor ya calificó a este pasajero en 'calificacion_pasajero'
        cal_doc = db["calificacion_pasajero"].find_one({"origen_tipo": "reserva", "id_registro": id_res}, {"_id": 0})
        tarjetas.append(
            {
                "origen_tipo": "reserva",
                "id_registro": id_res,
                "id_usuario": id_u,
                "nombre_pasajero": nombre_p,
                "inicial": nombre_p[0].upper() if nombre_p else "P",
                "matricula": "222310045" if id_u == 2 else f"2223100{id_u:02d}",
                "origen": origen,
                "destino": destino,
                "distancia_km": distancia,
                "tiempo_min": tiempo,
                "costo": int(costo) if costo.is_integer() else costo,
                "lugares": int(res.get("asientos_solicitados") or 1),
                "calificado": bool(cal_doc),
                "estrellas_dadas": cal_doc.get("estrellas") if cal_doc else None,
                "comentario_dado": cal_doc.get("comentario") if cal_doc else "",
                "puede_eliminar": False,
            }
        )

    # 2. Consultamos las rutas históricas de la colección 'ruta' (Abril Diaz, Blanca, Alexa, etc.)
    rutas_raw = list(db["ruta"].find({}, {"_id": 0}))
    for idx, r in enumerate(rutas_raw):
        id_ruta = int(r.get("id_ruta", idx + 1))
        demo = PASAJEROS_DEMO_RUTA[idx % len(PASAJEROS_DEMO_RUTA)]

        nombre_p = str(r.get("nombre_pasajero") or r.get("pasajero") or demo["nombre"]).strip()
        matricula = str(r.get("matricula") or demo["matricula"]).strip()
        id_u = int(r.get("id_usuario") or demo["id_usuario"])
        origen = r.get("origen") or r.get("punto_partida") or "Origen"
        destino = r.get("destino") or r.get("punto_llegada") or "Destino"

        # Rellenamos distancia, tiempo, tarifa y lugares si el documento histórico los trae en 0
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

        # Consultamos si esta ruta histórica ya fue calificada por el conductor
        cal_doc = db["calificacion_pasajero"].find_one({"origen_tipo": "ruta", "id_registro": id_ruta}, {"_id": 0})
        tarjetas.append(
            {
                "origen_tipo": "ruta",
                "id_registro": id_ruta,
                "id_usuario": id_u,
                "nombre_pasajero": nombre_p,
                "inicial": nombre_p[0].upper() if nombre_p else "P",
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

    # Renderizamos la vista HTML enviando la lista unificada y la bandera de aviso tras calificar
    return motor_plantillas.TemplateResponse(
        request=solicitud,
        name="ruta/ruta_inicio.html",
        context={
            "rutas": tarjetas,
            "aviso_calificado": bool(calificado),
        },
    )


@router.post("/calificar-pasajero")
def calificar_pasajero(
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
    """
    Ruta HTTP POST que guarda la calificación del conductor hacia un pasajero al finalizar el viaje.
    Actualiza 'calificacion_pasajero', la reserva/ruta origen y recalcula el promedio del pasajero.
    """
    # Validamos que el puntaje se mantenga estrictamente entre 1 y 5 estrellas
    estrellas_val = max(1, min(5, int(estrellas)))
    comentario_limpio = comentario.strip() or "Excelente pasajero, puntual y respetuoso durante el trayecto."
    fecha_actual = datetime.now().strftime("%Y-%m-%d %H:%M")

    # Evitamos duplicar calificaciones para el mismo viaje
    ya_calificado = db["calificacion_pasajero"].find_one(
        {"origen_tipo": origen_tipo, "id_registro": id_registro}
    )

    if not ya_calificado:
        # Generamos el siguiente ID autoincremental en la colección 'calificacion_pasajero'
        ultima = db["calificacion_pasajero"].find_one(sort=[("id_calificacion_pasajero", -1)])
        nuevo_id = (ultima["id_calificacion_pasajero"] + 1) if ultima and "id_calificacion_pasajero" in ultima else 1

        # Insertamos el documento de reseña que luego consultará el pasajero en su Historial
        doc = {
            "id_calificacion_pasajero": nuevo_id,
            "origen_tipo": origen_tipo,
            "id_registro": id_registro,
            "id_usuario": id_usuario,
            "nombre_pasajero": nombre_pasajero,
            "matricula_pasajero": matricula,
            "nombre_conductor": "Brandon Rivera",
            "origen": origen,
            "destino": destino,
            "estrellas": estrellas_val,
            "comentario": comentario_limpio,
            "fecha": fecha_actual,
        }
        db["calificacion_pasajero"].insert_one(doc)

        # Marcamos también la reserva o ruta original como ya calificada por el conductor
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

        # Recalculamos el promedio general del pasajero en la colección 'usuario'
        _recalcular_calificacion_pasajero(id_usuario, nombre_pasajero)

    # Redirigimos al Historial de Rutas mostrando el banner de confirmación
    return RedirectResponse(url="/ruta/?calificado=1", status_code=status.HTTP_303_SEE_OTHER)


@router.get("/nuevo", response_class=HTMLResponse)
def vista_crear_ruta(solicitud: Request):
    """Ruta HTTP GET que carga el formulario vacío para dar de alta una nueva ruta."""
    return motor_plantillas.TemplateResponse(
        request=solicitud,
        name="ruta/ruta_form.html",
        context={"ruta": None},
    )


@router.post("/guardar")
def guardar_ruta(
    origen: str = Form(...),
    destino: str = Form(...),
    distancia_km: float = Form(0.0),
    tiempo_min: int = Form(0),
    costo: float = Form(25.0),
):
    """Ruta HTTP POST que inserta una nueva ruta en la colección 'ruta' de MongoDB."""
    # Calculamos el siguiente identificador autoincremental
    ultima = db["ruta"].find_one(sort=[("id_ruta", -1)])
    nuevo_id = (ultima["id_ruta"] + 1) if ultima and "id_ruta" in ultima else 1

    # Guardamos el nuevo registro de ruta
    db["ruta"].insert_one(
        {
            "id_ruta": nuevo_id,
            "origen": origen,
            "destino": destino,
            "distancia_km": distancia_km,
            "tiempo_min": tiempo_min,
            "costo": costo,
            "lugares": 3,
        }
    )
    return RedirectResponse(url="/ruta/", status_code=status.HTTP_303_SEE_OTHER)


@router.get("/editar/{id_ruta}", response_class=HTMLResponse)
def vista_editar_ruta(solicitud: Request, id_ruta: int):
    """Ruta HTTP GET que busca una ruta por su ID y la carga en el formulario de edición."""
    ruta_doc = db["ruta"].find_one({"id_ruta": id_ruta}, {"_id": 0})
    return motor_plantillas.TemplateResponse(
        request=solicitud,
        name="ruta/ruta_form.html",
        context={"ruta": ruta_doc},
    )


@router.post("/actualizar")
def actualizar_ruta(
    id_ruta: int = Form(...),
    origen: str = Form(...),
    destino: str = Form(...),
    distancia_km: float = Form(0.0),
    tiempo_min: int = Form(0),
    costo: float = Form(25.0),
):
    """Ruta HTTP POST que actualiza los datos de una ruta existente en MongoDB."""
    db["ruta"].update_one(
        {"id_ruta": id_ruta},
        {
            "$set": {
                "origen": origen,
                "destino": destino,
                "distancia_km": distancia_km,
                "tiempo_min": tiempo_min,
                "costo": costo,
            }
        },
    )
    return RedirectResponse(url="/ruta/", status_code=status.HTTP_303_SEE_OTHER)


@router.get("/eliminar/{id_ruta}")
def eliminar_ruta(id_ruta: int):
    """Ruta HTTP GET que elimina una ruta histórica del historial."""
    db["ruta"].delete_one({"id_ruta": id_ruta})
    return RedirectResponse(url="/ruta/", status_code=status.HTTP_303_SEE_OTHER)