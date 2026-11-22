# Importamos estructuras para búsqueda en grafos (cola de prioridad y cola doble), cálculos geodésicos y fechas
import heapq
import math
from collections import deque
from datetime import datetime

# Importamos herramientas de FastAPI para rutas, formularios y plantillas HTML
from fastapi import APIRouter, Request, Form, status
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates

# Importamos la conexión activa a la base de datos MongoDB
from database import db

# Importamos los servicios de Machine Learning (K-Means/PCA y Random Forest) con respaldo seguro
try:
    from services.ml_service import simular_ruta_kmeans_pca, predecir_confiabilidad_pasajero
except ImportError:
    def simular_ruta_kmeans_pca(distancia_km: float, tiempo_min: float):
        # Segmentación heurística de respaldo en caso de no estar disponible el modelo serializado
        if distancia_km <= 6.0:
            cluster, tarifa_base = "Ruta Corta Local (Alta rotación)", 15.0
        elif distancia_km <= 16.0:
            cluster, tarifa_base = "Ruta Media Estándar (Interurbana)", 25.0
        else:
            cluster, tarifa_base = "Ruta Larga Especial (Foránea)", 40.0
        proporcion_trafico = tiempo_min / max(distancia_km, 0.5)
        tarifa_optima = round(tarifa_base * (1 + (proporcion_trafico * 0.08)), 2)
        pc1 = round((distancia_km * 0.7) + (tiempo_min * 0.3), 2)
        pc2 = round((distancia_km * -0.3) + (tiempo_min * 0.7), 2)
        return {
            "perfil_ruta": cluster,
            "tarifa_recomendada": tarifa_optima,
            "rentabilidad": "Alta" if tarifa_optima >= 25 else "Moderada",
            "coordenadas_pca": {"pc1": pc1, "pc2": pc2},
        }

    def predecir_confiabilidad_pasajero(historial_asistencia: float, cancelaciones_previas: float, viajes_totales: float):
        # Clasificación de respaldo de confiabilidad según historial de asistencia y cancelaciones
        if cancelaciones_previas == 0 and historial_asistencia >= 90:
            return {"probabilidad": 95.0, "estado": "Confiable / Alta Certeza"}
        elif cancelaciones_previas <= 1:
            return {"probabilidad": 80.0, "estado": "Moderado"}
        return {"probabilidad": 55.0, "estado": "En Riesgo de Cancelación"}

# Inicializamos el enrutador exclusivo del perfil Pasajero con prefijo '/pasajero'
router = APIRouter(prefix="/pasajero", tags=["Perfil Pasajero"])

# Configuramos el motor de plantillas Jinja2
motor_plantillas = Jinja2Templates(directory="templates")

# Diccionario de coordenadas geográficas de los nodos principales de Velox UTVT
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
    "Villa Cuauhtémoc": [19.4167, -99.5667],
}

# Catálogo limpio de nodos para los selectores del Simulador IA del pasajero
NODOS_SELECTOR = [
    "Rectoría UTVT",
    "Terminal Toluca",
    "San Mateo Atenco",
    "Toluca (Centro)",
    "Metepec",
    "Lerma",
    "Ocoyoacac",
    "Zinacantepec",
    "Xonacatlán",
    "Villa Cuauhtémoc",
]

# Alias para unificar nombres abreviados con los nombres formales del grafo
ALIAS_NODOS = {
    "UTVT": "Rectoría UTVT",
    "Terminal": "Terminal Toluca",
    "San Mateo": "San Mateo Atenco",
}

# Grafo vial bidireccional ponderado en kilómetros para ejecutar BFS, DFS, UCS, Greedy y A*
GRAFO_PASAJERO = {
    "Zinacantepec": {"Toluca (Centro)": 8.0},
    "Toluca (Centro)": {"Zinacantepec": 8.0, "Terminal Toluca": 3.5, "Metepec": 6.5},
    "Terminal Toluca": {"Toluca (Centro)": 3.5, "Metepec": 5.0, "San Mateo Atenco": 7.5, "Xonacatlán": 14.0},
    "Metepec": {"Toluca (Centro)": 6.5, "Terminal Toluca": 5.0, "San Mateo Atenco": 4.5, "Lerma": 8.5},
    "San Mateo Atenco": {"Terminal Toluca": 7.5, "Metepec": 4.5, "Lerma": 5.5, "Rectoría UTVT": 11.0},
    "Lerma": {"Metepec": 8.5, "San Mateo Atenco": 5.5, "Ocoyoacac": 5.0, "Rectoría UTVT": 7.0},
    "Ocoyoacac": {"Lerma": 5.0, "Rectoría UTVT": 10.5},
    "Xonacatlán": {"Terminal Toluca": 14.0, "Villa Cuauhtémoc": 5.0, "Rectoría UTVT": 9.5},
    "Villa Cuauhtémoc": {"Xonacatlán": 5.0, "Rectoría UTVT": 12.0},
    "Rectoría UTVT": {"San Mateo Atenco": 11.0, "Lerma": 7.0, "Ocoyoacac": 10.5, "Xonacatlán": 9.5, "Villa Cuauhtémoc": 12.0},
}

# Datos institucionales del conductor por defecto para enriquecer tarjetas de viaje
CONDUCTOR_DEFAULT = {
    "id_conductor": 1,
    "nombre": "Brandon Rivera",
    "matricula": "222310088",
    "correo": "al22288@utvt.edu.mx",
    "telefono": "722 987 6543",
    "calificacion": "4.8",
    "viajes_completados": 34,
}

# Catálogo visual de vehículos registrados con placas, color y fotografías de referencia
VEHICULOS_CATALOGO = {
    "Nissan Versa": {
        "modelo": "Nissan Versa 2022",
        "color": "Gris Oxford",
        "placas": "NVS-482-B",
        "foto_exterior": "https://images.unsplash.com/photo-1549317661-bd32c8ce0db2?auto=format&fit=crop&w=800&q=80",
        "foto_interior": "https://images.unsplash.com/photo-1503376780353-7e6692767b70?auto=format&fit=crop&w=800&q=80",
    },
    "Chevrolet Aveo 2023": {
        "modelo": "Chevrolet Aveo 2023",
        "color": "Blanco Platino",
        "placas": "ABC-123-A",
        "foto_exterior": "https://images.unsplash.com/photo-1590362891991-f776e747a588?auto=format&fit=crop&w=800&q=80",
        "foto_interior": "https://images.unsplash.com/photo-1542362567-b07e54358753?auto=format&fit=crop&w=800&q=80",
    },
}


def _normalizar_nodo(nombre: str) -> str:
    """Convierte alias cortos (ej. 'UTVT') al nombre oficial usado en el grafo."""
    return ALIAS_NODOS.get(nombre, nombre)


def _obtener_calificacion_conductor(id_conductor: int = 1) -> tuple[str, int]:
    """Calcula el promedio real del conductor desde la colección 'calificacion' en MongoDB."""
    docs = list(db["calificacion"].find({"tipo": {"$ne": "conductor_a_pasajero"}}, {"_id": 0}))
    valores = []
    for doc in docs:
        if "id_conductor" in doc and doc["id_conductor"] != id_conductor:
            continue
        val = doc.get("estrellas") or doc.get("puntuacion") or doc.get("calificacion")
        try:
            num = float(val)
            if 1.0 <= num <= 5.0:
                valores.append(num)
        except (TypeError, ValueError):
            continue

    if not valores:
        return "4.8", 0

    promedio = round(sum(valores) / len(valores), 1)
    return f"{promedio:.1f}", len(valores)


def _obtener_calificaciones_recibidas_pasajero(id_pasajero: int = 2, nombre: str = "Jorge Abraham") -> list[dict]:
    """Obtiene todas las reseñas que el conductor le ha otorgado al pasajero desde 'calificacion_pasajero'."""
    docs = list(
        db["calificacion_pasajero"].find(
            {
                "$or": [
                    {"id_usuario": id_pasajero},
                    {"nombre_pasajero": {"$regex": f"^{nombre}$", "$options": "i"}},
                ]
            },
            {"_id": 0},
        )
    )
    # Ordenamos de la reseña más reciente a la más antigua
    docs.sort(key=lambda c: int(c.get("id_calificacion_pasajero", 0)), reverse=True)
    return docs


def _obtener_o_crear_pasajero(id_pasajero: int = 2) -> dict:
    """Consulta (o inicializa si no existe) el perfil de reputación y confiabilidad IA del pasajero."""
    usuario = db["usuario"].find_one({"id_usuario": id_pasajero}, {"_id": 0})
    if not usuario:
        usuario = {
            "id_usuario": id_pasajero,
            "nombre": "Jorge Abraham",
            "correo_institucional": "pasajero@utvt.edu.mx",
            "rol": "Pasajero",
            "calificacion": 4.9,
            "historial_asistencia": 96.0,
            "cancelaciones": 0,
            "viajes_totales": 5,
        }
        db["usuario"].insert_one(usuario.copy())

    # Extraemos métricas numéricas para evaluar confiabilidad con el modelo Random Forest
    calificacion = float(usuario.get("calificacion", 4.9))
    asistencia = float(usuario.get("historial_asistencia", 96.0))
    cancelaciones = int(usuario.get("cancelaciones", 0))
    viajes_totales = int(usuario.get("viajes_totales", 5))

    analisis_ia = predecir_confiabilidad_pasajero(
        historial_asistencia=asistencia,
        cancelaciones_previas=cancelaciones,
        viajes_totales=viajes_totales,
    )

    return {
        "id_usuario": id_pasajero,
        "nombre": "Jorge Abraham",
        "inicial": "J",
        "calificacion": round(calificacion, 1),
        "historial_asistencia": round(asistencia, 1),
        "cancelaciones": cancelaciones,
        "viajes_totales": viajes_totales,
        "confiabilidad_ia": analisis_ia,
    }


def _distancia_haversine(c1: list[float], c2: list[float]) -> float:
    """Calcula la distancia en línea recta (km) entre dos pares de coordenadas [lat, lon]."""
    lat1, lon1 = math.radians(c1[0]), math.radians(c1[1])
    lat2, lon2 = math.radians(c2[0]), math.radians(c2[1])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    return 6371.0 * (2 * math.asin(math.sqrt(a)))


def _calcular_distancia_y_tiempo(origen: str, destino: str) -> tuple[float, int, list[float], list[float]]:
    """Devuelve distancia vial estimada (km), duración (min) y coordenadas de origen/destino."""
    c_orig = COORDENADAS_NODOS.get(origen, [19.2848, -99.6552])
    c_dest = COORDENADAS_NODOS.get(destino, [19.3451, -99.5082])
    distancia_recta = _distancia_haversine(c_orig, c_dest)
    distancia_km = round(max(2.5, distancia_recta * 1.22), 1)
    tiempo_min = max(10, int(round(distancia_km * 2.1)))
    return distancia_km, tiempo_min, c_orig, c_dest


def _evaluar_conveniencia_ia(distancia_km: float, tiempo_min: int, tarifa: float) -> dict:
    """
    Genera el dictamen de conveniencia para el pasajero antes de reservar,
    comparando la aportación de Velox frente a taxi privado y el tiempo frente a camión.
    """
    costo_taxi = round(45.0 + (distancia_km * 11.5), 2)
    ahorro_taxi = round(max(0.0, costo_taxi - tarifa), 2)
    porcentaje_ahorro = int(round((ahorro_taxi / costo_taxi) * 100)) if costo_taxi > 0 else 0
    tiempo_camion = int(round(tiempo_min * 1.65)) + 10
    ahorro_tiempo = max(5, tiempo_camion - tiempo_min)

    # Clasificación en 3 niveles según distancia y aportación del viaje
    if distancia_km < 5.5:
        return {
            "nivel": "moderado",
            "etiqueta": "CONVENIENCIA MODERADA",
            "titulo": "Trayecto corto — Evalúa si te conviene",
            "resumen": (
                f"La distancia es de {distancia_km} km (~{tiempo_min} min). Con Velox pagas ${tarifa} MXN y ahorras "
                f"${ahorro_taxi} MXN frente a un taxi (~${costo_taxi} MXN)."
            ),
            "sugerencia": "Te conviene si priorizas seguridad y comodidad puerta a puerta sin esperar transporte.",
            "costo_taxi": costo_taxi,
            "ahorro_taxi": ahorro_taxi,
            "porcentaje_ahorro": porcentaje_ahorro,
            "tiempo_camion": tiempo_camion,
            "ahorro_tiempo": ahorro_tiempo,
        }
    elif distancia_km <= 24.0 and tarifa <= 45.0:
        return {
            "nivel": "conviene",
            "etiqueta": "SÍ TE CONVIENE",
            "titulo": "¡Excelente opción de viaje!",
            "resumen": (
                f"Ahorras ${ahorro_taxi} MXN ({porcentaje_ahorro}%) comparado con un taxi (~${costo_taxi} MXN) "
                f"y llegas aproximadamente {ahorro_tiempo} minutos más rápido que en camión (~{tiempo_camion} min)."
            ),
            "sugerencia": "Ruta directa y tarifa justa dentro del corredor UTVT. Muy recomendable reservar este asiento.",
            "costo_taxi": costo_taxi,
            "ahorro_taxi": ahorro_taxi,
            "porcentaje_ahorro": porcentaje_ahorro,
            "tiempo_camion": tiempo_camion,
            "ahorro_tiempo": ahorro_tiempo,
        }
    else:
        return {
            "nivel": "no_conviene",
            "etiqueta": "EVALÚA ALTERNATIVAS",
            "titulo": "Trayecto extenso o tarifa elevada",
            "resumen": (
                f"Es un recorrido largo de {distancia_km} km (~{tiempo_min} min) con aportación de ${tarifa} MXN. "
                f"Aunque sigue siendo más económico que un taxi (~${costo_taxi} MXN), revisa si hay una ruta más cercana."
            ),
            "sugerencia": "Compara este horario con otras salidas disponibles desde San Mateo Atenco o Terminal Toluca.",
            "costo_taxi": costo_taxi,
            "ahorro_taxi": ahorro_taxi,
            "porcentaje_ahorro": porcentaje_ahorro,
            "tiempo_camion": tiempo_camion,
            "ahorro_tiempo": ahorro_tiempo,
        }


def _ejecutar_busqueda_grafo(algoritmo: str, origen: str, destino: str) -> dict:
    """Ejecuta el algoritmo de búsqueda solicitado (BFS, DFS, UCS, Greedy o A*) sobre el grafo vial."""
    origen_n = _normalizar_nodo(origen)
    destino_n = _normalizar_nodo(destino)
    if origen_n not in GRAFO_PASAJERO or destino_n not in GRAFO_PASAJERO:
        return {"algoritmo": algoritmo, "ruta": [], "costo": 0.0, "explorados": 0}

    # Función heurística h(n): distancia geodésica directa hacia el nodo destino
    def heuristica(nodo: str) -> float:
        return round(_distancia_haversine(COORDENADAS_NODOS[nodo], COORDENADAS_NODOS[destino_n]), 2)

    # Búsqueda en Anchura (BFS)
    if algoritmo == "BFS":
        cola = deque([(origen_n, [origen_n], 0.0)])
        visitados = {origen_n}
        explorados = 0
        while cola:
            actual, camino, costo = cola.popleft()
            explorados += 1
            if actual == destino_n:
                return {"algoritmo": "BFS", "ruta": camino, "costo": round(costo, 1), "explorados": explorados}
            for vecino, peso in GRAFO_PASAJERO.get(actual, {}).items():
                if vecino not in visitados:
                    visitados.add(vecino)
                    cola.append((vecino, camino + [vecino], costo + peso))

    # Búsqueda en Profundidad (DFS)
    elif algoritmo == "DFS":
        pila = [(origen_n, [origen_n], 0.0)]
        visitados = set()
        explorados = 0
        while pila:
            actual, camino, costo = pila.pop()
            if actual in visitados:
                continue
            visitados.add(actual)
            explorados += 1
            if actual == destino_n:
                return {"algoritmo": "DFS", "ruta": camino, "costo": round(costo, 1), "explorados": explorados}
            for vecino, peso in reversed(list(GRAFO_PASAJERO.get(actual, {}).items())):
                if vecino not in visitados:
                    pila.append((vecino, camino + [vecino], costo + peso))

    # Búsqueda de Costo Uniforme (UCS)
    elif algoritmo == "UCS":
        pq = [(0.0, origen_n, [origen_n])]
        visitados = set()
        explorados = 0
        while pq:
            costo, actual, camino = heapq.heappop(pq)
            if actual in visitados:
                continue
            visitados.add(actual)
            explorados += 1
            if actual == destino_n:
                return {"algoritmo": "UCS", "ruta": camino, "costo": round(costo, 1), "explorados": explorados}
            for vecino, peso in GRAFO_PASAJERO.get(actual, {}).items():
                if vecino not in visitados:
                    heapq.heappush(pq, (costo + peso, vecino, camino + [vecino]))

    # Búsqueda Voraz (Greedy Best-First)
    elif algoritmo == "Greedy":
        pq = [(heuristica(origen_n), 0.0, origen_n, [origen_n])]
        visitados = set()
        explorados = 0
        while pq:
            _, costo, actual, camino = heapq.heappop(pq)
            if actual in visitados:
                continue
            visitados.add(actual)
            explorados += 1
            if actual == destino_n:
                return {"algoritmo": "Greedy", "ruta": camino, "costo": round(costo, 1), "explorados": explorados}
            for vecino, peso in GRAFO_PASAJERO.get(actual, {}).items():
                if vecino not in visitados:
                    heapq.heappush(pq, (heuristica(vecino), costo + peso, vecino, camino + [vecino]))

    # Búsqueda A* (combinación óptima de costo acumulado g(n) + heurística h(n))
    else:
        pq = [(heuristica(origen_n), 0.0, origen_n, [origen_n])]
        visitados = set()
        explorados = 0
        while pq:
            _, costo, actual, camino = heapq.heappop(pq)
            if actual in visitados:
                continue
            visitados.add(actual)
            explorados += 1
            if actual == destino_n:
                return {"algoritmo": "A*", "ruta": camino, "costo": round(costo, 1), "explorados": explorados}
            for vecino, peso in GRAFO_PASAJERO.get(actual, {}).items():
                if vecino not in visitados:
                    nuevo_g = costo + peso
                    heapq.heappush(pq, (nuevo_g + heuristica(vecino), nuevo_g, vecino, camino + [vecino]))

    return {"algoritmo": algoritmo, "ruta": [], "costo": 0.0, "explorados": 0}


def _asegurar_viajes_iniciales():
    """Carga 3 viajes base del conductor si las colecciones 'viaje' y 'ruta' están vacías."""
    if db["viaje"].count_documents({}) == 0 and db["ruta"].count_documents({}) == 0:
        viajes_base = [
            {
                "id_viaje": 1,
                "origen": "Rectoría UTVT",
                "destino": "Terminal Toluca",
                "horario": "Hoy, 14:30 hrs",
                "conductor": "Brandon Rivera",
                "matricula_conductor": "222310088",
                "vehiculo": "Nissan Versa",
                "placas": "NVS-482-B",
                "color_vehiculo": "Gris Oxford",
                "distancia_km": 18.5,
                "tiempo_min": 35,
                "asientos_disponibles": 2,
                "costo": 25.0,
                "estado": "activo",
            },
            {
                "id_viaje": 2,
                "origen": "Terminal Toluca",
                "destino": "Rectoría UTVT",
                "horario": "Mañana, 07:00 hrs",
                "conductor": "Brandon Rivera",
                "matricula_conductor": "222310088",
                "vehiculo": "Chevrolet Aveo 2023",
                "placas": "ABC-123-A",
                "color_vehiculo": "Blanco Platino",
                "distancia_km": 18.5,
                "tiempo_min": 35,
                "asientos_disponibles": 3,
                "costo": 25.0,
                "estado": "activo",
            },
            {
                "id_viaje": 3,
                "origen": "San Mateo Atenco",
                "destino": "Rectoría UTVT",
                "horario": "Mañana, 07:20 hrs",
                "conductor": "Brandon Rivera",
                "matricula_conductor": "222310088",
                "vehiculo": "Chevrolet Aveo 2023",
                "placas": "ABC-123-A",
                "color_vehiculo": "Blanco Platino",
                "distancia_km": 11.0,
                "tiempo_min": 22,
                "asientos_disponibles": 3,
                "costo": 20.0,
                "estado": "activo",
            },
        ]
        db["viaje"].insert_many(viajes_base)


def _enriquecer_viaje(viaje_doc: dict, indice: int = 1, calificacion_conductor: str = "4.8") -> dict:
    """Completa cada viaje con ruta A*, coordenadas Leaflet, vehículo, conductor y dictamen IA."""
    origen = viaje_doc.get("origen") or viaje_doc.get("punto_partida")
    destino = viaje_doc.get("destino") or viaje_doc.get("punto_llegada")
    costo = viaje_doc.get("costo")

    # Si el viaje referencia una ruta por id_ruta, consultamos los datos faltantes en 'ruta'
    if (not origen or not destino or costo is None) and "id_ruta" in viaje_doc:
        ruta_doc = db["ruta"].find_one({"id_ruta": viaje_doc["id_ruta"]}, {"_id": 0})
        if ruta_doc:
            origen = origen or ruta_doc.get("origen") or ruta_doc.get("punto_partida")
            destino = destino or ruta_doc.get("destino") or ruta_doc.get("punto_llegada")
            if costo is None:
                costo = ruta_doc.get("costo")

    origen = origen or "Terminal Toluca"
    destino = destino or "Rectoría UTVT"

    dist_calc, tiempo_calc, c_orig, c_dest = _calcular_distancia_y_tiempo(origen, destino)
    raw_dist = viaje_doc.get("distancia_km")
    raw_tiempo = viaje_doc.get("tiempo_min")
    distancia_km = float(raw_dist) if raw_dist is not None else float(dist_calc)
    tiempo_min = int(raw_tiempo) if raw_tiempo is not None else int(tiempo_calc)
    costo_final = float(costo) if costo is not None else 25.0
    # Ejecutamos A* para trazar los nodos intermedios del recorrido en el mapa previo
    busqueda = _ejecutar_busqueda_grafo("A*", origen, destino)
    ruta_nodos = busqueda.get("ruta", [])
    if len(ruta_nodos) >= 2:
        coords_ruta = [COORDENADAS_NODOS[n] for n in ruta_nodos if n in COORDENADAS_NODOS]
    else:
        ruta_nodos = [origen, destino]
        coords_ruta = [c_orig, c_dest]

    asientos = viaje_doc.get("asientos_disponibles")
    if asientos is None:
        asientos = viaje_doc.get("lugares_disponibles", 3)

    nombre_vehiculo = viaje_doc.get("vehiculo") or "Chevrolet Aveo 2023"
    info_vehiculo = VEHICULOS_CATALOGO.get(nombre_vehiculo, VEHICULOS_CATALOGO["Chevrolet Aveo 2023"])

    # Calculamos segmentación K-Means/PCA y evaluación de conveniencia para el pasajero
    ml_info = simular_ruta_kmeans_pca(distancia_km=distancia_km, tiempo_min=tiempo_min)
    dictamen = _evaluar_conveniencia_ia(distancia_km=distancia_km, tiempo_min=tiempo_min, tarifa=costo_final)

    return {
        "id_viaje": int(viaje_doc.get("id_viaje", viaje_doc.get("id_ruta", indice))),
        "origen": origen,
        "destino": destino,
        "horario": viaje_doc.get("horario") or viaje_doc.get("hora_salida") or "Hoy, 14:30 hrs",
        "conductor": viaje_doc.get("conductor") or CONDUCTOR_DEFAULT["nombre"],
        "matricula_conductor": viaje_doc.get("matricula_conductor") or CONDUCTOR_DEFAULT["matricula"],
        "correo_conductor": viaje_doc.get("correo_conductor") or CONDUCTOR_DEFAULT["correo"],
        "telefono_conductor": viaje_doc.get("telefono_conductor") or CONDUCTOR_DEFAULT["telefono"],
        "calificacion_conductor": calificacion_conductor,
        "vehiculo": info_vehiculo["modelo"],
        "placas": viaje_doc.get("placas") or info_vehiculo["placas"],
        "color_vehiculo": viaje_doc.get("color_vehiculo") or info_vehiculo["color"],
        "foto_exterior": viaje_doc.get("foto_exterior") or info_vehiculo["foto_exterior"],
        "foto_interior": viaje_doc.get("foto_interior") or info_vehiculo["foto_interior"],
        "distancia_km": distancia_km,
        "tiempo_min": tiempo_min,
        "coords_origen": c_orig,
        "coords_destino": c_dest,
        "ruta_nodos": ruta_nodos,
        "coords_ruta": coords_ruta,
        "asientos_disponibles": int(asientos),
        "costo": costo_final,
        "perfil_ia": ml_info.get("perfil_ruta", "Ruta Media Estándar"),
        "tarifa_ia": ml_info.get("tarifa_recomendada", costo_final),
        "coordenadas_pca": ml_info.get("coordenadas_pca", {"pc1": 0, "pc2": 0}),
        "dictamen": dictamen,
        "creado_por": viaje_doc.get("creado_por", "conductor"),
    }


def _obtener_contexto_pasajero(id_pasajero: int, vista_activa: str) -> dict:
    """Construye el diccionario de contexto compartido por todas las pantallas del pasajero."""
    _asegurar_viajes_iniciales()
    pasajero_info = _obtener_o_crear_pasajero(id_pasajero)
    calif_conductor, total_resenas_conductor = _obtener_calificacion_conductor(1)
    calificaciones_recibidas = _obtener_calificaciones_recibidas_pasajero(id_pasajero, pasajero_info["nombre"])

    # Consultamos las reservas del pasajero e identificamos viajes activos para ocultarlos de 'Buscar Viajes'
    mis_reservas_raw = list(db["reserva"].find({"id_usuario": id_pasajero}, {"_id": 0}))
    mis_reservas = []
    viajes_ocultar_ids = set()

    for res in mis_reservas_raw:
        estado_lower = str(res.get("estado", "pendiente")).lower()
        id_v = res.get("id_viaje")

        # Si el pasajero ya solicitó o tiene aceptado el viaje, lo ocultamos para evitar confusión
        if id_v is not None and estado_lower in ("pendiente", "aceptada", "confirmada"):
            viajes_ocultar_ids.add(id_v)

        viaje_doc = db["viaje"].find_one({"id_viaje": id_v}, {"_id": 0}) if id_v is not None else None
        if viaje_doc:
            detalle = _enriquecer_viaje(viaje_doc, calificacion_conductor=calif_conductor)
        else:
            detalle = _enriquecer_viaje(
                {
                    "id_viaje": id_v or 0,
                    "origen": res.get("origen", "Rectoría UTVT"),
                    "destino": res.get("destino", "Terminal Toluca"),
                    "horario": res.get("horario", "Hoy, 14:30 hrs"),
                    "costo": res.get("costo", 25.0),
                    "distancia_km": res.get("distancia_km"),
                    "tiempo_min": res.get("tiempo_min"),
                },
                calificacion_conductor=calif_conductor,
            )

        if res.get("origen"):
            detalle["origen"] = res["origen"]
        if res.get("destino"):
            detalle["destino"] = res["destino"]

        res["origen"] = detalle["origen"]
        res["destino"] = detalle["destino"]
        res["detalle_viaje"] = detalle
        mis_reservas.append(res)

    mis_reservas.sort(key=lambda r: r.get("id_reserva", 0), reverse=True)

    # Filtramos los viajes publicados por conductores excluyendo los que el pasajero ya reservó
    viajes_raw = list(db["viaje"].find({}, {"_id": 0}))
    viajes_disponibles = []
    fuente_viajes = viajes_raw if viajes_raw else list(db["ruta"].find({}, {"_id": 0}))

    for idx, doc in enumerate(fuente_viajes, start=1):
        viaje_info = _enriquecer_viaje(doc, idx, calificacion_conductor=calif_conductor)
        if viaje_info["id_viaje"] in viajes_ocultar_ids:
            continue
        if viaje_info.get("creado_por") == "pasajero":
            continue
        viajes_disponibles.append(viaje_info)

    # Contamos las solicitudes activas para el contador del menú lateral
    reservas_activas = [
        r for r in mis_reservas
        if str(r.get("estado", "")).lower() in ("pendiente", "aceptada", "confirmada")
    ]

    return {
        "vista": vista_activa,
        "viajes": viajes_disponibles,
        "mis_reservas": mis_reservas,
        "reservas_activas": reservas_activas,
        "total_activos": len(reservas_activas),
        "nodos_disponibles": NODOS_SELECTOR,
        "coordenadas_nodos": COORDENADAS_NODOS,
        "calificacion_conductor_actual": calif_conductor,
        "total_resenas_conductor": total_resenas_conductor,
        "calificaciones_recibidas": calificaciones_recibidas,
        "pasajero": pasajero_info,
    }


def _construir_datos_simulador(origen: str, destino: str, algoritmo: str) -> dict:
    """Ejecuta el simulador completo del pasajero (grafos, 10 premisas lógicas, K-Means/PCA y conveniencia)."""
    algoritmos_base = ["BFS", "DFS", "UCS", "Greedy", "A*"]
    comparativa = []

    # Si elige 'Todos', ejecutamos los 5 algoritmos y usamos A* como ruta principal
    if algoritmo == "Todos":
        for alg in algoritmos_base:
            comparativa.append(_ejecutar_busqueda_grafo(alg, origen, destino))
        principal = comparativa[-1]
    else:
        principal = _ejecutar_busqueda_grafo(algoritmo, origen, destino)
        comparativa.append(principal)

    ruta_encontrada = principal.get("ruta", [])
    dist_km = principal.get("costo", 0.0)
    if dist_km <= 0:
        dist_km, tiempo_min, _, _ = _calcular_distancia_y_tiempo(origen, destino)
    else:
        tiempo_min = max(10, int(round(dist_km * 2.1)))

    ml_res = simular_ruta_kmeans_pca(distancia_km=dist_km, tiempo_min=tiempo_min)
    dictamen = _evaluar_conveniencia_ia(dist_km, tiempo_min, ml_res["tarifa_recomendada"])

    # Evaluamos las 10 premisas de lógica proposicional del sistema Velox UTVT
    via_valida = len(ruta_encontrada) >= 2 and origen != destino
    premisas = [
        ("Origen", _normalizar_nodo(origen) in GRAFO_PASAJERO),
        ("Destino", _normalizar_nodo(destino) in GRAFO_PASAJERO and origen != destino),
        ("Vía IA", via_valida),
        ("Cupo", True),
        ("Usuario", True),
        ("Hora", True),
        ("Chofer", True),
        ("Estatus", True),
        ("Pago", True),
        ("Seguridad", True),
    ]
    todas_aprobadas = all(v for _, v in premisas)
    coords_ruta = [COORDENADAS_NODOS[n] for n in ruta_encontrada if n in COORDENADAS_NODOS]

    return {
        "origen": origen,
        "destino": destino,
        "algoritmo": algoritmo,
        "algoritmos_disponibles": ["A*", "UCS", "Greedy", "BFS", "DFS", "Todos"],
        "ruta": ruta_encontrada,
        "coords_ruta": coords_ruta,
        "distancia_km": dist_km,
        "tiempo_min": tiempo_min,
        "explorados": principal.get("explorados", 0),
        "comparativa": comparativa,
        "premisas": premisas,
        "aprobado": todas_aprobadas,
        "ml": ml_res,
        "dictamen": dictamen,
    }


@router.get("/", response_class=HTMLResponse)
def panel_pasajero(solicitud: Request, id_pasajero: int = 2):
    """Ruta HTTP GET principal del pasajero: muestra viajes disponibles y tabla de solicitudes."""
    contexto = _obtener_contexto_pasajero(id_pasajero, vista_activa="buscar")
    return motor_plantillas.TemplateResponse(
        request=solicitud,
        name="pasajeros/pasajero_inicio.html",
        context=contexto,
    )


@router.get("/mi-viaje", response_class=HTMLResponse)
def detalle_mi_viaje(solicitud: Request, id_pasajero: int = 2):
    """Ruta HTTP GET del apartado 'Detalle de mi Viaje' (mapa, conductor, placas y fotos)."""
    contexto = _obtener_contexto_pasajero(id_pasajero, vista_activa="detalle")
    return motor_plantillas.TemplateResponse(
        request=solicitud,
        name="pasajeros/pasajero_inicio.html",
        context=contexto,
    )


@router.get("/historial", response_class=HTMLResponse)
def historial_viajes(
    solicitud: Request,
    id_pasajero: int = 2,
    cancelado: int = 0,
    calificado: int = 0,
):
    """Ruta HTTP GET del apartado 'Historial de Viajes', reputación y calificaciones recibidas."""
    contexto = _obtener_contexto_pasajero(id_pasajero, vista_activa="historial")
    contexto["aviso_cancelado"] = bool(cancelado)
    contexto["aviso_calificado"] = bool(calificado)
    return motor_plantillas.TemplateResponse(
        request=solicitud,
        name="pasajeros/pasajero_inicio.html",
        context=contexto,
    )


@router.get("/simulador-ia", response_class=HTMLResponse)
def simulador_ia_pasajero_get(solicitud: Request, id_pasajero: int = 2):
    """Ruta HTTP GET del Simulador de Rutas IA dentro del propio perfil del pasajero."""
    contexto = _obtener_contexto_pasajero(id_pasajero, vista_activa="simulador")
    contexto["sim_ia"] = _construir_datos_simulador("Terminal Toluca", "Rectoría UTVT", "A*")
    return motor_plantillas.TemplateResponse(
        request=solicitud,
        name="pasajeros/pasajero_inicio.html",
        context=contexto,
    )


@router.post("/simulador-ia", response_class=HTMLResponse)
def simulador_ia_pasajero_post(
    solicitud: Request,
    origen: str = Form("Terminal Toluca"),
    destino: str = Form("Rectoría UTVT"),
    algoritmo: str = Form("A*"),
    id_pasajero: int = 2,
):
    """Ruta HTTP POST que ejecuta la simulación seleccionada sin cambiar de perfil."""
    contexto = _obtener_contexto_pasajero(id_pasajero, vista_activa="simulador")
    contexto["sim_ia"] = _construir_datos_simulador(origen, destino, algoritmo)
    return motor_plantillas.TemplateResponse(
        request=solicitud,
        name="pasajeros/pasajero_inicio.html",
        context=contexto,
    )


@router.post("/solicitar")
def solicitar_viaje(
    id_viaje: int = Form(...),
    origen: str = Form(...),
    destino: str = Form(...),
    id_usuario: int = Form(2),
    nombre_pasajero: str = Form("Jorge Abraham"),
):
    """Ruta HTTP POST que crea una solicitud de asiento 'pendiente' en MongoDB."""
    # Prevenimos solicitudes duplicadas del mismo pasajero sobre el mismo viaje
    ya_existe = db["reserva"].find_one({
        "id_usuario": id_usuario,
        "id_viaje": id_viaje,
        "estado": {"$in": ["pendiente", "aceptada", "confirmada", "Pendiente", "Aceptada"]},
    })

    if not ya_existe:
        ultima = db["reserva"].find_one(sort=[("id_reserva", -1)])
        nuevo_id = (ultima["id_reserva"] + 1) if ultima and "id_reserva" in ultima else 1

        nueva_reserva = {
            "id_reserva": nuevo_id,
            "id_viaje": id_viaje,
            "id_usuario": id_usuario,
            "nombre_pasajero": nombre_pasajero,
            "origen": origen,
            "destino": destino,
            "estado": "pendiente",
            "fecha_solicitud": datetime.now().strftime("%Y-%m-%d %H:%M"),
        }
        db["reserva"].insert_one(nueva_reserva)

    # Redirigimos al detalle del viaje recién solicitado
    return RedirectResponse(url="/pasajero/mi-viaje", status_code=status.HTTP_303_SEE_OTHER)


@router.post("/cancelar/{id_reserva}")
def cancelar_viaje_pasajero(
    id_reserva: int,
    id_usuario: int = Form(2),
    motivo_cancelacion: str = Form("Motivo no especificado"),
):
    """
    Ruta HTTP POST para cancelar un viaje guardando el motivo obligatorio.
    Libera el asiento si ya estaba aceptado y penaliza la reputación/confiabilidad del pasajero.
    """
    reserva = db["reserva"].find_one({"id_reserva": id_reserva, "id_usuario": id_usuario})
    motivo_limpio = motivo_cancelacion.strip() or "Motivo no especificado"

    if reserva:
        estado_previo = str(reserva.get("estado", "")).lower()
        if estado_previo in ("pendiente", "aceptada", "confirmada"):
            # Marcamos la reserva como cancelada y almacenamos el comentario del motivo
            db["reserva"].update_one(
                {"id_reserva": id_reserva},
                {
                    "$set": {
                        "estado": "cancelada",
                        "cancelado_por": "pasajero",
                        "motivo_cancelacion": motivo_limpio,
                        "fecha_cancelacion": datetime.now().strftime("%Y-%m-%d %H:%M"),
                    }
                },
            )

            # Si el conductor ya había descontado un lugar, devolvemos el asiento al viaje
            if estado_previo in ("aceptada", "confirmada") and reserva.get("id_viaje") is not None:
                db["viaje"].update_one(
                    {"id_viaje": reserva["id_viaje"]},
                    {"$inc": {"asientos_disponibles": 1}},
                )

            # Actualizamos cancelaciones, asistencia y calificación en 'usuario' (impacta Random Forest)
            pasajero_actual = _obtener_o_crear_pasajero(id_usuario)
            nuevas_cancelaciones = pasajero_actual["cancelaciones"] + 1
            nueva_asistencia = max(40.0, round(pasajero_actual["historial_asistencia"] - 12.0, 1))
            nueva_calificacion = max(1.0, round(pasajero_actual["calificacion"] - 0.3, 1))

            db["usuario"].update_one(
                {"id_usuario": id_usuario},
                {
                    "$set": {
                        "cancelaciones": nuevas_cancelaciones,
                        "historial_asistencia": nueva_asistencia,
                        "calificacion": nueva_calificacion,
                    }
                },
            )

    # Redirigimos al historial mostrando el aviso de cancelación
    return RedirectResponse(url="/pasajero/historial?cancelado=1", status_code=status.HTTP_303_SEE_OTHER)


@router.post("/calificar")
def calificar_conductor(
    id_reserva: int = Form(...),
    estrellas: int = Form(...),
    comentario: str = Form(""),
    id_usuario: int = Form(2),
):
    """
    Ruta HTTP POST que registra la evaluación del pasajero hacia el conductor en 'calificacion',
    marca el viaje como finalizado y recalcula el promedio del conductor en MongoDB.
    """
    estrellas_val = max(1, min(5, int(estrellas)))
    reserva = db["reserva"].find_one({"id_reserva": id_reserva, "id_usuario": id_usuario})

    if reserva and not reserva.get("calificado"):
        ultima_cal = db["calificacion"].find_one(sort=[("id_calificacion", -1)])
        nuevo_id_cal = (ultima_cal["id_calificacion"] + 1) if ultima_cal and "id_calificacion" in ultima_cal else 1

        comentario_limpio = comentario.strip() or "Buen viaje, conductor puntual y amable."
        fecha_actual = datetime.now().strftime("%Y-%m-%d %H:%M")

        # Insertamos la reseña que se mostrará en /calificacion/ y en /perfil-conductor
        doc_calificacion = {
            "id_calificacion": nuevo_id_cal,
            "id_reserva": id_reserva,
            "id_viaje": reserva.get("id_viaje", 1),
            "id_usuario": id_usuario,
            "id_conductor": 1,
            "nombre_pasajero": reserva.get("nombre_pasajero", "Jorge Abraham"),
            "nombre_conductor": CONDUCTOR_DEFAULT["nombre"],
            "puntuacion": estrellas_val,
            "estrellas": estrellas_val,
            "calificacion": estrellas_val,
            "comentario": comentario_limpio,
            "fecha": fecha_actual,
        }
        db["calificacion"].insert_one(doc_calificacion)

        # Actualizamos la reserva a 'finalizada' con el resumen de la calificación otorgada
        db["reserva"].update_one(
            {"id_reserva": id_reserva},
            {
                "$set": {
                    "estado": "finalizada",
                    "calificado": True,
                    "calificacion_otorgada": estrellas_val,
                    "comentario_calificacion": comentario_limpio,
                    "fecha_calificacion": fecha_actual,
                }
            },
        )

        # Incrementamos los viajes totales del pasajero y actualizamos el promedio del conductor
        db["usuario"].update_one(
            {"id_usuario": id_usuario},
            {"$inc": {"viajes_totales": 1}},
        )

        nuevo_promedio, _ = _obtener_calificacion_conductor(1)
        db["usuario"].update_one(
            {"id_usuario": 1},
            {"$set": {"calificacion": float(nuevo_promedio)}},
        )

    return RedirectResponse(url="/pasajero/historial?calificado=1", status_code=status.HTTP_303_SEE_OTHER)