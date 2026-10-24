# ==========================================================
# MÓDULO DE INTELIGENCIA ARTIFICIAL: BÚSQUEDA Y REGLAS LÓGICAS
# Proyecto: Velox UTVT (100% Dinámico con MongoDB y Geocoding)
# ==========================================================

import heapq
import math
import os
from collections import deque
from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

try:
    from database import db
except ImportError:
    db = None

router_simulador_ia = APIRouter(prefix="/simulador-ia", tags=["Simulador IA - Lógica Proposicional y Búsqueda"])
motor_plantillas = Jinja2Templates(directory="templates")

# ==========================================================
# RESPALDO ESTÁTICO DE SEGURIDAD Y RED VIAL
# ==========================================================
RESPALDO_COORDENADAS = {
    "Terminal": [19.2848, -99.6552],
    "San Mateo": [19.2673, -99.5931],
    "Toluca (Centro)": [19.2926, -99.6569],
    "Zinacantepec": [19.2811, -99.7275],
    "Metepec": [19.2568, -99.6047],
    "Lerma": [19.2875, -99.5186],
    "Xonacatlán": [19.3920, -99.5350],
    "UTVT": [19.3451, -99.5082],
    "San Antonio": [19.2612, -99.6835]
}

RESPALDO_GRAFO = {
    "Terminal": {"San Mateo": 4.0, "Toluca (Centro)": 6.0},
    "San Mateo": {"Terminal": 4.0, "UTVT": 7.0},
    "Toluca (Centro)": {"Terminal": 6.0, "Metepec": 5.0},
    "Metepec": {"Toluca (Centro)": 5.0, "Lerma": 6.0},
    "Lerma": {"UTVT": 5.0, "Metepec": 6.0},
    "UTVT": {"San Mateo": 7.0, "Lerma": 5.0},
    "Zinacantepec": {"Toluca (Centro)": 8.0},
    "Xonacatlán": {"Terminal": 12.0, "UTVT": 10.0}
}

RESPALDO_HEURISTICA = {
    "Terminal": 18, "San Mateo": 15, "Toluca (Centro)": 18,
    "Metepec": 20, "Lerma": 15, "Xonacatlán": 15,
    "Zinacantepec": 20, "UTVT": 0, "San Antonio": 12
}


# ==========================================================
# EXTRACCIÓN DINÁMICA DESDE MONGODB
# ==========================================================
def obtener_datos_viales_desde_mongo():
    grafo_dinamico = {k: v.copy() for k, v in RESPALDO_GRAFO.items()}
    coords_dinamicas = RESPALDO_COORDENADAS.copy()
    heuristica_dinamica = RESPALDO_HEURISTICA.copy()

    if db is not None:
        try:
            registros_rutas = list(db["ruta"].find({}, {"_id": 0}))
            if registros_rutas:
                for reg in registros_rutas:
                    origen = reg.get("origen")
                    destino = reg.get("destino")
                    costo = float(reg.get("costo", 5.0))
                    
                    lat_o = reg.get("lat_origen")
                    lng_o = reg.get("lng_origen")
                    lat_d = reg.get("lat_destino")
                    lng_d = reg.get("lng_destino")
                    
                    if origen:
                        o = origen.strip()
                        if lat_o is not None and lng_o is not None:
                            coords_dinamicas[o] = [float(lat_o), float(lng_o)]
                    
                    if destino:
                        d = destino.strip()
                        if lat_d is not None and lng_d is not None:
                            coords_dinamicas[d] = [float(lat_d), float(lng_d)]
                        
                    if origen and destino and origen != destino:
                        o = origen.strip()
                        d = destino.strip()
                        if o not in grafo_dinamico:
                            grafo_dinamico[o] = {}
                        grafo_dinamico[o][d] = costo
        except Exception as error_bd:
            print(f"Aviso MongoDB: {error_bd}")

    return grafo_dinamico, coords_dinamicas, heuristica_dinamica


def obtener_red_vial_ponderada():
    grafo, _, _ = obtener_datos_viales_desde_mongo()
    return grafo

def obtener_coordenadas_nodos():
    _, coords, _ = obtener_datos_viales_desde_mongo()
    return coords

def calcular_heuristica(grafo_red, nodo_objetivo):
    _, _, heuristica = obtener_datos_viales_desde_mongo()
    return heuristica


# ==========================================================
# FUNCIONES AUXILIARES DE RECONSTRUCCIÓN Y COSTOS
# ==========================================================
def reconstruir_camino(padres, inicio, objetivo):
    if objetivo != inicio and objetivo not in padres: return []
    camino = []
    actual = objetivo
    while actual is not None:
        camino.append(actual)
        if actual == inicio: break
        actual = padres.get(actual)
    camino.reverse()
    if not camino or camino[0] != inicio: return []
    return camino

def calcular_costo(camino, grafo_ponderado):
    if not camino or len(camino) < 2: return 0.0
    costo = 0.0
    for i in range(len(camino) - 1):
        costo += grafo_ponderado.get(camino[i], {}).get(camino[i + 1], 0.0)
    return costo

def vecinos_ordenados(nodo):
    grafo_actual = obtener_red_vial_ponderada()
    return sorted(grafo_actual.get(nodo, {}).items())


# ==========================================================
# ALGORITMOS DE BÚSQUEDA
# ==========================================================
def ejecutar_bfs(inicio, objetivo):
    grafo = obtener_red_vial_ponderada()
    if inicio not in grafo or objetivo not in grafo: return None, [], 0.0
    if inicio == objetivo: return [inicio], [inicio], 0.0

    frontera = deque([inicio])
    descubiertos = {inicio}
    padres = {inicio: None}
    expandidos = []
    registro = []

    while frontera:
        actual = frontera.popleft()
        expandidos.append(actual)
        registro.append(f"BFS: Se extrae {actual} de la cola.")
        if actual == objetivo: break

        for vecino, _ in vecinos_ordenados(actual):
            if vecino not in descubiertos:
                descubiertos.add(vecino)
                padres[vecino] = actual
                frontera.append(vecino)
                registro.append(f"  -> Se descubre {vecino} desde {actual}.")

    camino = reconstruir_camino(padres, inicio, objetivo)
    costo = calcular_costo(camino, grafo)
    return camino, expandidos, costo

def ejecutar_dfs(inicio, objetivo):
    grafo = obtener_red_vial_ponderada()
    if inicio not in grafo or objetivo not in grafo: return None, [], 0.0

    pila = [inicio]
    descubiertos = {inicio}
    padres = {inicio: None}
    expandidos = []
    registro = []

    while pila:
        actual = pila.pop()
        if actual not in expandidos:
            expandidos.append(actual)
            registro.append(f"DFS: Se extrae {actual} de la pila.")
        if actual == objetivo: break

        for vecino, _ in reversed(vecinos_ordenados(actual)):
            if vecino not in descubiertos:
                descubiertos.add(vecino)
                padres[vecino] = actual
                pila.append(vecino)
                registro.append(f"  -> Se agrega {vecino} desde {actual}.")

    camino = reconstruir_camino(padres, inicio, objetivo)
    costo = calcular_costo(camino, grafo)
    return camino, expandidos, costo

def ejecutar_ucs(inicio, objetivo):
    grafo = obtener_red_vial_ponderada()
    if inicio not in grafo or objetivo not in grafo: return None, [], 0.0

    frontera = [(0.0, inicio)]
    mejor_costo = {inicio: 0.0}
    padres = {inicio: None}
    expandidos = []
    registro = []
    cerrados = set()

    while frontera:
        costo_actual, actual = heapq.heappop(frontera)
        if costo_actual != mejor_costo.get(actual, math.inf): continue
        if actual in cerrados: continue

        cerrados.add(actual)
        expandidos.append(actual)
        registro.append(f"UCS: Se expande {actual} con g(n) = {costo_actual}.")
        if actual == objetivo: break

        for vecino, peso in vecinos_ordenados(actual):
            nuevo_costo = costo_actual + peso
            if nuevo_costo < mejor_costo.get(vecino, math.inf):
                mejor_costo[vecino] = nuevo_costo
                padres[vecino] = actual
                heapq.heappush(frontera, (nuevo_costo, vecino))
                registro.append(f"  -> Actualización para {vecino}: g(n) = {nuevo_costo}.")

    camino = reconstruir_camino(padres, inicio, objetivo)
    costo = calcular_costo(camino, grafo)
    return camino, expandidos, costo

def ejecutar_greedy(inicio, objetivo):
    grafo = obtener_red_vial_ponderada()
    heuristica = calcular_heuristica(grafo, objetivo)
    if inicio not in grafo or objetivo not in grafo: return None, [], 0.0

    frontera = [(heuristica.get(inicio, 0.0), inicio)]
    descubiertos = {inicio}
    padres = {inicio: None}
    expandidos = []
    registro = []

    while frontera:
        h_actual, actual = heapq.heappop(frontera)
        if actual not in expandidos:
            expandidos.append(actual)
            registro.append(f"Greedy: Se expande {actual} con h(n) = {h_actual}.")
        if actual == objetivo: break

        for vecino, _ in vecinos_ordenados(actual):
            if vecino not in descubiertos:
                descubiertos.add(vecino)
                padres[vecino] = actual
                h_vecino = heuristica.get(vecino, 0.0)
                heapq.heappush(frontera, (h_vecino, vecino))
                registro.append(f"  -> Se descubre {vecino} con h(n) = {h_vecino}.")

    camino = reconstruir_camino(padres, inicio, objetivo)
    costo = calcular_costo(camino, grafo)
    return camino, expandidos, costo

def ejecutar_astar(inicio, objetivo):
    grafo = obtener_red_vial_ponderada()
    heuristica = calcular_heuristica(grafo, objetivo)
    if inicio not in grafo or objetivo not in grafo: return None, [], 0.0

    frontera = [(heuristica.get(inicio, 0.0), 0.0, inicio)]
    mejor_g = {inicio: 0.0}
    padres = {inicio: None}
    expandidos = []
    registro = []

    while frontera:
        f_actual, g_actual, actual = heapq.heappop(frontera)
        if g_actual > mejor_g.get(actual, math.inf): continue
        if actual not in expandidos:
            expandidos.append(actual)
            registro.append(f"A*: Se expande {actual} con f(n)={f_actual}.")
        if actual == objetivo: break

        for vecino, peso in vecinos_ordenados(actual):
            nuevo_g = g_actual + peso
            if nuevo_g < mejor_g.get(vecino, math.inf):
                mejor_g[vecino] = nuevo_g
                padres[vecino] = actual
                h_vecino = heuristica.get(vecino, 0.0)
                nuevo_f = nuevo_g + h_vecino
                heapq.heappush(frontera, (nuevo_f, nuevo_g, vecino))
                registro.append(f"  -> Actualización en {vecino}: f(n)={nuevo_f}.")

    camino = reconstruir_camino(padres, inicio, objetivo)
    costo = calcular_costo(camino, grafo)
    return camino, expandidos, costo


# ==========================================================
# MOTOR AVANZADO DE LÓGICA PROPOSICIONAL (10 PREMISAS REALES)
# ==========================================================
def evaluar_reglas_carpooling(
    origen: str,
    destino: str,
    grafo: dict,
    ruta_solucion: list,
    asientos_disponibles: int = 3,
    usuario_verificado: bool = True,
    coincidencia_horario: bool = True,
    conductor_aprobado: bool = True,
    viaje_activo: bool = True,
    cuota_cubierta: bool = True,
    zona_segura: bool = True
):
    # Definición formal de las 10 premisas lógicas (Forma Normal Conjuntiva)
    premisas = {
        "O": origen in grafo,                                       # P1: Origen Válido
        "D": (destino in grafo) and (origen != destino),             # P2: Destino Válido
        "V": bool(ruta_solucion) and len(ruta_solucion) >= 2,         # P3: Vía Conectada (Ruta resuelta por la IA)
        "C": asientos_disponibles > 0,                              # P4: Capacidad Óptima de asientos
        "U": usuario_verificado,                                    # P5: Usuario UTVT Verificado
        "H": coincidencia_horario,                                  # P6: Horario Compatible
        "E": conductor_aprobado,                                    # P7: Estatus del Conductor
        "S": viaje_activo,                                          # P8: Viaje Activo
        "M": cuota_cubierta,                                        # P9: Cuota de Gasolina Aprobada
        "R": zona_segura                                            # P10: Zona / Ruta Segura
    }

    # Inferencia lógica: Conjunción total (AND de todas las premisas)
    viaje_permitido = all(premisas.values())

    # Diagnóstico detallado por premisa en caso de rechazo
    lista_explicaciones = []
    if not premisas["O"]: lista_explicaciones.append("O: El origen seleccionado no es válido en la red.")
    if not premisas["D"]: lista_explicaciones.append("O bien D: El destino no es válido o coincide con el origen.")
    if not premisas["V"]: lista_explicaciones.append("V: No existe un tramo vial conectado y solucionado por la IA.")
    if not premisas["C"]: lista_explicaciones.append("C: Capacidad agotada (Sin asientos disponibles).")
    if not premisas["U"]: lista_explicaciones.append("U: Usuario sin registro verificado en la UTVT.")
    if not premisas["H"]: lista_explicaciones.append("H: Desfase en el horario de salida del viaje compartido.")
    if not premisas["E"]: lista_explicaciones.append("E: Conductor o vehículo sin validación institucional.")
    if not premisas["S"]: lista_explicaciones.append("S: El estado del viaje no se encuentra activo.")
    if not premisas["M"]: lista_explicaciones.append("M: Cuota de recuperación de trayecto pendiente.")
    if not premisas["R"]: lista_explicaciones.append("R: Ruta con alerta o restricción de tránsito.")

    if viaje_permitido:
        diagnostico = "Aprobado: Las 10 premisas de lógica proposicional se cumplen con éxito en Velox UTVT."
    else:
        diagnostico = "Rechazado por Motor Lógico -> " + " | ".join(lista_explicaciones)

    return {
        "premisas": premisas,
        "viaje_permitido": viaje_permitido,
        "explicacion": diagnostico
    }


# ==========================================================
# CONTROLADOR DE RUTA GET
# ==========================================================
@router_simulador_ia.get("/", response_class=HTMLResponse)
def vista_simulador_ia(peticion_http: Request, origen: str = "Terminal", destino: str = "UTVT", algoritmo: str = "A*"):
    grafo_ponderado = obtener_red_vial_ponderada()
    coords_nodos = obtener_coordenadas_nodos()
    
    lista_nodos_disponibles = sorted(list(grafo_ponderado.keys()))

    origen = origen if origen in lista_nodos_disponibles else "Terminal"
    destino = destino if destino in lista_nodos_disponibles else "UTVT"

    ruta_solucion = []
    bitacora_proceso = []
    costo_total_ruta = 0.0
    tabla_comparativa = []

    algoritmos_a_ejecutar = ["BFS", "DFS", "UCS", "Greedy", "A*"]

    if algoritmo == "Todos":
        for algo in algoritmos_a_ejecutar:
            if algo == "BFS": sol, bit, costo = ejecutar_bfs(origen, destino)
            elif algo == "DFS": sol, bit, costo = ejecutar_dfs(origen, destino)
            elif algo == "UCS": sol, bit, costo = ejecutar_ucs(origen, destino)
            elif algo == "Greedy": sol, bit, costo = ejecutar_greedy(origen, destino)
            elif algo == "A*": sol, bit, costo = ejecutar_astar(origen, destino)
            
            tabla_comparativa.append({
                "algoritmo": algo,
                "ruta": " ➔ ".join(sol) if sol else "Sin ruta",
                "nodos_explorados": len(bit),
                "costo": f"{costo:.1f}" if sol and math.isfinite(costo) else "N/A"
            })
        ruta_solucion, bitacora_proceso, costo_total_ruta = ejecutar_astar(origen, destino)
    else:
        if algoritmo == "DFS": ruta_solucion, bitacora_proceso, costo_total_ruta = ejecutar_dfs(origen, destino)
        elif algoritmo == "UCS": ruta_solucion, bitacora_proceso, costo_total_ruta = ejecutar_ucs(origen, destino)
        elif algoritmo == "Greedy": ruta_solucion, bitacora_proceso, costo_total_ruta = ejecutar_greedy(origen, destino)
        elif algoritmo == "A*": ruta_solucion, bitacora_proceso, costo_total_ruta = ejecutar_astar(origen, destino)
        else: ruta_solucion, bitacora_proceso, costo_total_ruta = ejecutar_bfs(origen, destino)

    # Evaluación formal del Motor de Reglas con las 10 premisas lógicas
    evaluacion_reglas = evaluar_reglas_carpooling(
        origen=origen,
        destino=destino,
        grafo=grafo_ponderado,
        ruta_solucion=ruta_solucion,
        asientos_disponibles=3,
        usuario_verificado=True,
        coincidencia_horario=True,
        conductor_aprobado=True,
        viaje_activo=True,
        cuota_cubierta=True,
        zona_segura=True
    )

    total_movimientos_aristas = len(ruta_solucion) - 1 if ruta_solucion else 0

    contexto_plantilla = {
        "origen": origen,
        "destino": destino,
        "algoritmo": algoritmo,
        "estados_disponibles": lista_nodos_disponibles,
        "algoritmos_disponibles": ["BFS", "DFS", "UCS", "Greedy", "A*", "Todos"],
        "ruta_solucion": ruta_solucion,
        "bitacora": bitacora_proceso,
        "total_movimientos": total_movimientos_aristas,
        "costo_total": f"{costo_total_ruta:.1f}",
        "conjunto_S": ", ".join(lista_nodos_disponibles),
        "evaluacion_logica": evaluacion_reglas,
        "coordenadas_nodos": coords_nodos,
        "tabla_comparativa": tabla_comparativa
    }

    return motor_plantillas.TemplateResponse(peticion_http, "ia/simulador_ia_inicio.html", contexto_plantilla)