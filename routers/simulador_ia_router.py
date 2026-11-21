# ==========================================================
# MÓDULO DE INTELIGENCIA ARTIFICIAL: BÚSQUEDA, REGLAS Y ML
# Proyecto: Velox UTVT (100% Dinámico con MongoDB y Geocoding)
# ==========================================================

import heapq
import math
import os
import joblib
import numpy as np
from collections import deque
from fastapi import APIRouter, Request, HTTPException, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

# Importamos las funciones de nuestro nuevo servicio de IA (K-Means y PCA)
from services.ml_service import simular_ruta_kmeans_pca

try:
    from database import db
except ImportError:
    db = None

router_simulador_ia = APIRouter(prefix="/simulador-ia", tags=["Simulador IA - Módulo Completo"])
motor_plantillas = Jinja2Templates(directory="templates")

# ==========================================================
# CARGA DEL MODELO DE MACHINE LEARNING (KNN)
# ==========================================================
RUTA_MODELO = os.path.join("..", "ia_models", "modelo_velox.pkl")
try:
    # Se carga el cerebro predictivo exportado en la práctica anterior
    modelo_ml = joblib.load(RUTA_MODELO)
    print("✅ Modelo de Machine Learning cargado exitosamente en el backend.")
except Exception as e:
    modelo_ml = None
    print(f"⚠️ Aviso: No se pudo cargar el modelo ML. ¿Está en {RUTA_MODELO}? Error: {e}")

# Esquema para recibir los 4 datos que requiere el modelo
class DatosPrediccionML(BaseModel):
    # Adaptamos las variables originales (horas_estudio, asistencia, etc.) al contexto de viaje
    clima: float          # Ej. 0 a 20
    trafico: float        # Ej. 40 a 100
    estado_auto: float    # Ej. 0 a 100
    experiencia: float    # Ej. 0 a 10


# ==========================================================
# RESPALDO ESTÁTICO DE SEGURIDAD Y RED VIAL
# ==========================================================
RESPALDO_COORDENADAS = {
    "Terminal": [19.2848, -99.6552], "San Mateo": [19.2673, -99.5931],
    "Toluca (Centro)": [19.2926, -99.6569], "Zinacantepec": [19.2811, -99.7275],
    "Metepec": [19.2568, -99.6047], "Lerma": [19.2875, -99.5186],
    "Xonacatlán": [19.3920, -99.5350], "Villa Cuauhtemoc": [19.4167, -99.5667],
    "UTVT": [19.3451, -99.5082], "San Antonio": [19.2612, -99.6835]
}

RESPALDO_GRAFO = {
    "Terminal": {"San Mateo": 4.0, "Toluca (Centro)": 6.0},
    "San Mateo": {"Terminal": 4.0, "UTVT": 7.0},
    "Toluca (Centro)": {"Terminal": 6.0, "Metepec": 5.0},
    "Metepec": {"Toluca (Centro)": 5.0, "Lerma": 6.0},
    "Lerma": {"UTVT": 5.0, "Metepec": 6.0},
    "UTVT": {"San Mateo": 7.0, "Lerma": 5.0},
    "Villa Cuauhtemoc": {"Xonacatlán": 5.0, "UTVT": 12.0},
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
            registro.append(f"A*: Se expande {actual} with f(n)={f_actual}.")
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
def evaluar_reglas_carpooling(origen: str, destino: str, grafo: dict, ruta_solucion: list, asientos_disponibles: int = 3, usuario_verificado: bool = True, coincidencia_horario: bool = True, conductor_aprobado: bool = True, viaje_activo: bool = True, cuota_cubierta: bool = True, zona_segura: bool = True):
    premisas = {
        "O": origen in grafo,
        "D": (destino in grafo) and (origen != destino),
        "V": bool(ruta_solucion) and len(ruta_solucion) >= 2,
        "C": asientos_disponibles > 0,
        "U": usuario_verificado,
        "H": coincidencia_horario,
        "E": conductor_aprobado,
        "S": viaje_activo,
        "M": cuota_cubierta,
        "R": zona_segura
    }

    viaje_permitido = all(premisas.values())
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
# ENDPOINT GET: VISTA DEL SIMULADOR IA (BÚSQUEDA Y LÓGICA)
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

    evaluacion_reglas = evaluar_reglas_carpooling(
        origen=origen, destino=destino, grafo=grafo_ponderado, ruta_solucion=ruta_solucion,
        asientos_disponibles=3, usuario_verificado=True, coincidencia_horario=True,
        conductor_aprobado=True, viaje_activo=True, cuota_cubierta=True, zona_segura=True
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


# ==========================================================
# ENDPOINT POST: PREDICCIÓN MACHINE LEARNING (KNN)
# ==========================================================
@router_simulador_ia.post("/api/predecir")
def predecir_ruta_ml(datos: DatosPrediccionML):
    """
    Este endpoint consume el modelo físico (.pkl) generado con Scikit-Learn.
    Recibe las 4 características del entorno y devuelve si el viaje es viable (1) o no (0).
    """
    if modelo_ml is None:
        raise HTTPException(status_code=500, detail="Modelo ML no cargado. Verifica la carpeta ia_models.")
    
    try:
        # Convertimos los datos JSON al array matemático 2D que Scikit-Learn espera
        entrada = np.array([[datos.clima, datos.trafico, datos.estado_auto, datos.experiencia]])
        
        # El modelo procesa los datos y escupe una predicción
        prediccion = modelo_ml.predict(entrada)
        resultado_binario = int(prediccion[0])
        
        # Mapeamos el resultado matemático a un contexto real para Velox UTVT
        if resultado_binario == 1:
            diagnostico = "Ruta Óptima y Segura (El estudiante llega a tiempo)"
        else:
            diagnostico = "Ruta con Riesgo / Probable Retraso"
            
        return {
            "exito": True,
            "prediccion": resultado_binario,
            "diagnostico": diagnostico
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ==========================================================
# ENDPOINT POST: ANÁLISIS DE RUTAS (K-MEANS & PCA)
# ==========================================================
@router_simulador_ia.post("/analizar", response_class=HTMLResponse)
def analizar_ruta_kmeans_pca(
    solicitud: Request, 
    distancia: float = Form(...), 
    tiempo: float = Form(...)
):
    """
    Procesa los datos de un trayecto y devuelve el análisis 
    de K-Means (Segmentación de Tarifas) y PCA (Variables Dimensionales).
    """
    # 1. Llamamos a la lógica matemática en nuestro servicio externo
    analisis_resultados = simular_ruta_kmeans_pca(distancia_km=distancia, tiempo_min=tiempo)
    
    # 2. Reconstruimos los datos base de los grafos para que la vista HTML 
    # no se rompa al intentar renderizar el mapa o los selectores si están en la misma página.
    grafo_ponderado = obtener_red_vial_ponderada()
    coords_nodos = obtener_coordenadas_nodos()
    lista_nodos_disponibles = sorted(list(grafo_ponderado.keys()))

    contexto = {
        # Nuevos datos de K-Means y PCA
        "resultado": analisis_resultados,
        "distancia": distancia,
        "tiempo": tiempo,
        "usuario": {"nombre": "Brandon Rivera"},
        
        # Datos del mapa base para evitar errores de renderizado
        "estados_disponibles": lista_nodos_disponibles,
        "algoritmos_disponibles": ["BFS", "DFS", "UCS", "Greedy", "A*", "Todos"],
        "coordenadas_nodos": coords_nodos,
        "origen": "Terminal", 
        "destino": "UTVT"
    }

    # Asumimos que integraste ambos diseños en "ia/simulador_ia_inicio.html".
    # (Si los dejaste en plantillas separadas, cambia esta ruta por "simulador/simulador_inicio.html")
    return motor_plantillas.TemplateResponse(solicitud, "ia/simulador_ia_inicio.html", contexto)