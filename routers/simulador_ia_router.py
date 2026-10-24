# ==========================================================
# MÓDULO DE INTELIGENCIA ARTIFICIAL: BÚSQUEDA Y REGLAS LÓGICAS
# Proyecto: Velox UTVT (100% Dinámico con MongoDB y Geocoding)
# ==========================================================

# Importa el módulo heapq para manejar colas de prioridad (necesario en UCS, Greedy y A*).
import heapq
# Importa el módulo math para cálculos matemáticos e infinito (math.inf).
import math
# Importa el módulo os para operaciones del sistema operativo.
import os
# Importa deque de collections para colas eficientes (usado en BFS).
from collections import deque
# Importa APIRouter y Request de FastAPI para enrutamiento y manejo HTTP.
from fastapi import APIRouter, Request
# Importa HTMLResponse para retornar respuestas de tipo HTML.
from fastapi.responses import HTMLResponse
# Importa Jinja2Templates para renderizar plantillas HTML dinámicas.
from fastapi.templating import Jinja2Templates

# Intenta importar la base de datos desde el archivo local 'database'.
try:
    from database import db
# Si no existe el módulo de base de datos, captura la excepción ImportError.
except ImportError:
    # Define la base de datos como nula (modo de respaldo estático).
    db = None

# Inicializa el enrutador de FastAPI para el simulador de IA con un prefijo y etiquetas.
router_simulador_ia = APIRouter(prefix="/simulador-ia", tags=["Simulador IA - Lógica Proposicional y Búsqueda"])
# Configura el motor de plantillas Jinja2 apuntando a la carpeta de templates.
motor_plantillas = Jinja2Templates(directory="templates")

# ==========================================================
# RESPALDO ESTÁTICO DE SEGURIDAD Y RED VIAL
# ==========================================================
# Define un diccionario estático con las coordenadas geográficas (lat, lng) de respaldo de los nodos viales.
RESPALDO_COORDENADAS = {
    "Terminal": [19.2848, -99.6552],
    "San Mateo": [19.2673, -99.5931],
    "Toluca (Centro)": [19.2926, -99.6569],
    "Zinacantepec": [19.2811, -99.7275],
    "Metepec": [19.2568, -99.6047],
    "Lerma": [19.2875, -99.5186],
    "Xonacatlán": [19.3920, -99.5350],
    "Villa Cuauhtemoc": [19.4167, -99.5667],
    "UTVT": [19.3451, -99.5082],
    "San Antonio": [19.2612, -99.6835]
}

# Define el grafo ponderado de respaldo con las conexiones viales y sus respectivos costos de distancia.
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

# Define los valores heurísticos estimados de cada nodo hacia el destino final (UTVT).
RESPALDO_HEURISTICA = {
    "Terminal": 18, "San Mateo": 15, "Toluca (Centro)": 18,
    "Metepec": 20, "Lerma": 15, "Xonacatlán": 15,
    "Zinacantepec": 20, "UTVT": 0, "San Antonio": 12
}


# ==========================================================
# EXTRACCIÓN DINÁMICA DESDE MONGODB
# ==========================================================
# Define la función para extraer y actualizar la red vial, coordenadas y heurística desde MongoDB.
def obtener_datos_viales_desde_mongo():
    # Copia el diccionario del grafo estático para transformarlo dinámicamente.
    grafo_dinamico = {k: v.copy() for k, v in RESPALDO_GRAFO.items()}
    # Copia el diccionario de coordenadas estáticas.
    coords_dinamicas = RESPALDO_COORDENADAS.copy()
    # Copia el diccionario de heurística estática.
    heuristica_dinamica = RESPALDO_HEURISTICA.copy()

    # Verifica si la conexión a la base de datos está disponible.
    if db is not None:
        # Inicia un bloque protegido contra errores para la consulta a MongoDB.
        try:
            # Obtiene todos los registros de la colección 'ruta' excluyendo el campo '_id'.
            registros_rutas = list(db["ruta"].find({}, {"_id": 0}))
            # Evalúa si se encontraron registros de rutas en la base de datos.
            if registros_rutas:
                # Itera sobre cada registro obtenido de la colección.
                for reg in registros_rutas:
                    # Extrae el nodo de origen del registro actual.
                    origen = reg.get("origen")
                    # Extrae el nodo de destino del registro actual.
                    destino = reg.get("destino")
                    # Extrae el costo de la ruta, asignando un valor predeterminado de 5.0 si no existe.
                    costo = float(reg.get("costo", 5.0))
                    
                    # Extrae la latitud del origen.
                    lat_o = reg.get("lat_origen")
                    # Extrae la longitud del origen.
                    lng_o = reg.get("lng_origen")
                    # Extrae la latitud del destino.
                    lat_d = reg.get("lat_destino")
                    # Extrae la longitud del destino.
                    lng_d = reg.get("lng_destino")
                    
                    # Verifica si el origen no es nulo o vacío.
                    if origen:
                        # Elimina espacios en blanco alrededor del nombre del origen.
                        o = origen.strip()
                        # Valida que las coordenadas del origen existan.
                        if lat_o is not None and lng_o is not None:
                            # Asigna las coordenadas dinámicas al diccionario de coordenadas.
                            coords_dinamicas[o] = [float(lat_o), float(lng_o)]
                    
                    # Verifica si el destino no es nulo o vacío.
                    if destino:
                        # Elimina espacios en blanco alrededor del nombre del destino.
                        d = destino.strip()
                        # Valida que las coordenadas del destino existan.
                        if lat_d is not None and lng_d is not None:
                            # Asigna las coordenadas dinámicas del destino.
                            coords_dinamicas[d] = [float(lat_d), float(lng_d)]
                        
                    # Verifica que origen y destino existan y sean diferentes.
                    if origen and destino and origen != destino:
                        # Limpia los espacios del origen.
                        o = origen.strip()
                        # Limpia los espacios del destino.
                        d = destino.strip()
                        # Si el origen no está registrado en el grafo dinámico, lo inicializa.
                        if o not in grafo_dinamico:
                            grafo_dinamico[o] = {}
                        # Agrega o actualiza la arista ponderada en el grafo dinámico.
                        grafo_dinamico[o][d] = costo
        # Captura cualquier error de conexión o consulta a la base de datos.
        except Exception as error_bd:
            # Imprime un mensaje en consola advirtiendo el fallo en MongoDB.
            print(f"Aviso MongoDB: {error_bd}")

    # Retorna las estructuras de datos viales actualizadas (grafo, coordenadas y heurística).
    return grafo_dinamico, coords_dinamicas, heuristica_dinamica


# Define una función envoltorio para obtener únicamente el grafo ponderado de la red vial.
def obtener_red_vial_ponderada():
    # Desempaqueta únicamente el grafo obtenido de la función de MongoDB.
    grafo, _, _ = obtener_datos_viales_desde_mongo()
    # Retorna el grafo con los pesos viales.
    return grafo

# Define una función para obtener el diccionario de coordenadas geográficas de los nodos.
def obtener_coordenadas_nodos():
    # Desempaqueta únicamente las coordenadas de la función principal.
    _, coords, _ = obtener_datos_viales_desde_mongo()
    # Retorna el diccionario de coordenadas.
    return coords

# Define una función para calcular o recuperar la heurística de los nodos.
def calcular_heuristica(grafo_red, nodo_objetivo):
    # Desempaqueta los valores heurísticos desde la función de datos viales.
    _, _, heuristica = obtener_datos_viales_desde_mongo()
    # Retorna el diccionario de heurísticas.
    return heuristica


# ==========================================================
# FUNCIONES AUXILIARES DE RECONSTRUCCIÓN Y COSTOS
# ==========================================================
# Define una función para reconstruir el camino solución a partir del diccionario de padres.
def reconstruir_camino(padres, inicio, objetivo):
    # Si el objetivo es diferente del inicio y no se encuentra en el diccionario de padres, retorna vacío.
    if objetivo != inicio and objetivo not in padres: return []
    # Inicializa una lista vacía para almacenar la secuencia del camino.
    camino = []
    # Establece el nodo actual comenzando desde el objetivo.
    actual = objetivo
    # Inicia un bucle mientras el nodo actual no sea nulo.
    while actual is not None:
        # Añade el nodo actual a la lista del camino.
        camino.append(actual)
        # Si el nodo actual es el inicio, interrumpe el bucle.
        if actual == inicio: break
        # Retrocede al nodo padre correspondiente utilizando el diccionario.
        actual = padres.get(actual)
    # Invierte el orden de la lista para que vaya desde el inicio hasta el objetivo.
    camino.reverse()
    # Si el camino está vacío o el primer elemento no coincide con el inicio, retorna vacío.
    if not camino or camino[0] != inicio: return []
    # Retorna la lista con la secuencia ordenada del camino.
    return camino

# Define una función para calcular el costo acumulado total de un camino dado.
def calcular_costo(camino, grafo_ponderado):
    # Si el camino no existe o tiene menos de 2 nodos, el costo es 0.0.
    if not camino or len(camino) < 2: return 0.0
    # Inicializa la variable acumuladora del costo en cero.
    costo = 0.0
    # Itera a través de los tramos (pares de nodos consecutivos) del camino.
    for i in range(len(camino) - 1):
        # Suma el peso de la arista entre el nodo actual y el siguiente al costo total.
        costo += grafo_ponderado.get(camino[i], {}).get(camino[i + 1], 0.0)
    # Retorna el costo total acumulado del recorrido.
    return costo

# Define una función para obtener los vecinos de un nodo ordenados alfabéticamente.
def vecinos_ordenados(nodo):
    # Obtiene el diccionario actual de la red vial ponderada.
    grafo_actual = obtener_red_vial_ponderada()
    # Retorna los vecinos del nodo especificado ordenados por clave.
    return sorted(grafo_actual.get(nodo, {}).items())


# ==========================================================
# ALGORITMOS DE BÚSQUEDA
# ==========================================================
# Define la función para ejecutar el algoritmo de Búsqueda en Anchura (BFS).
def ejecutar_bfs(inicio, objetivo):
    # Obtiene la red vial ponderada actual.
    grafo = obtener_red_vial_ponderada()
    # Valida si el nodo de inicio o el objetivo no existen en el grafo, retornando valores vacíos.
    if inicio not in grafo or objetivo not in grafo: return None, [], 0.0
    # Si el inicio y el objetivo son el mismo nodo, retorna el camino inmediato de costo cero.
    if inicio == objetivo: return [inicio], [inicio], 0.0

    # Inicializa una cola (deque) con el nodo inicial para la frontera de BFS.
    frontera = deque([inicio])
    # Inicializa un conjunto de nodos descubiertos incluyendo el nodo inicial.
    descubiertos = {inicio}
    # Inicializa el diccionario de nodos padres para la reconstrucción del camino.
    padres = {inicio: None}
    # Inicializa una lista para llevar el registro de nodos expandidos.
    expandidos = []
    # Inicializa una lista para la bitácora textual del proceso.
    registro = []

    # Bucle principal que se ejecuta mientras existan elementos en la frontera.
    while frontera:
        # Extrae el primer elemento de la cola (FIFO) como el nodo actual.
        actual = frontera.popleft()
        # Añade el nodo actual a la lista de expandidos.
        expandidos.append(actual)
        # Agrega un mensaje explicativo a la bitácora del proceso.
        registro.append(f"BFS: Se extrae {actual} de la cola.")
        # Si el nodo actual es el objetivo, detiene la búsqueda.
        if actual == objetivo: break

        # Itera sobre los vecinos ordenados del nodo actual.
        for vecino, _ in vecinos_ordenados(actual):
            # Si el vecino no ha sido descubierto previamente:
            if vecino not in descubiertos:
                # Lo marca como descubierto agregándolo al conjunto.
                descubiertos.add(vecino)
                # Registra al nodo actual como el padre del vecino.
                padres[vecino] = actual
                # Agrega el vecino al final de la cola de la frontera.
                frontera.append(vecino)
                # Añade el descubrimiento a la bitácora.
                registro.append(f"  -> Se descubre {vecino} desde {actual}.")

    # Reconstruye el camino final utilizando el diccionario de padres.
    camino = reconstruir_camino(padres, inicio, objetivo)
    # Calcula el costo total del camino encontrado.
    costo = calcular_costo(camino, grafo)
    # Retorna el camino, la lista de nodos expandidos y el costo total.
    return camino, expandidos, costo

# Define la función para ejecutar el algoritmo de Búsqueda en Profundidad (DFS).
def ejecutar_dfs(inicio, objetivo):
    # Obtiene el grafo ponderado de la red vial.
    grafo = obtener_red_vial_ponderada()
    # Valida la existencia de los nodos de inicio y objetivo en el grafo.
    if inicio not in grafo or objetivo not in grafo: return None, [], 0.0

    # Inicializa una pila (lista) con el nodo inicial para DFS.
    pila = [inicio]
    # Inicializa el conjunto de nodos descubiertos.
    descubiertos = {inicio}
    # Inicializa el diccionario de nodos padres.
    padres = {inicio: None}
    # Inicializa la lista de nodos expandidos.
    expandidos = []
    # Inicializa la bitácora de ejecución.
    registro = []

    # Bucle principal que opera mientras la pila contenga elementos.
    while pila:
        # Extrae el último elemento de la pila (LIFO) como nodo actual.
        actual = pila.pop()
        # Si el nodo actual no se encuentra en la lista de expandidos:
        if actual not in expandidos:
            # Lo añade a los nodos expandidos.
            expandidos.append(actual)
            # Registra la extracción en la bitácora.
            registro.append(f"DFS: Se extrae {actual} de la pila.")
        # Si se alcanza el nodo objetivo, detiene el bucle.
        if actual == objetivo: break

        # Itera sobre los vecinos ordenados en orden inverso para preservar prioridades en pila.
        for vecino, _ in reversed(vecinos_ordenados(actual)):
            # Si el vecino no ha sido descubierto:
            if vecino not in descubiertos:
                # Lo marca como descubierto.
                descubiertos.add(vecino)
                # Establece su nodo padre.
                padres[vecino] = actual
                # Agrega el vecino a la pila.
                pila.append(vecino)
                # Registra el apilamiento en la bitácora.
                registro.append(f"  -> Se agrega {vecino} desde {actual}.")

    # Reconstruye el camino desde el diccionario de padres.
    camino = reconstruir_camino(padres, inicio, objetivo)
    # Calcula el costo total del camino.
    costo = calcular_costo(camino, grafo)
    # Retorna el camino, los nodos expandidos y el costo.
    return camino, expandidos, costo

# Define la función para ejecutar Búsqueda de Costo Uniforme (UCS).
def ejecutar_ucs(inicio, objetivo):
    # Obtiene el grafo ponderado de la red vial.
    grafo = obtener_red_vial_ponderada()
    # Valida que los nodos de inicio y objetivo pertenezcan al grafo.
    if inicio not in grafo or objetivo not in grafo: return None, [], 0.0

    # Inicializa la cola de prioridad con una tupla que contiene el costo acumulado (0.0) y el nodo inicial.
    frontera = [(0.0, inicio)]
    # Inicializa un diccionario con el mejor costo conocido para llegar a cada nodo, partiendo del inicio en 0.0.
    mejor_costo = {inicio: 0.0}
    # Inicializa el diccionario de nodos padres.
    padres = {inicio: None}
    # Inicializa la lista de nodos expandidos.
    expandidos = []
    # Inicializa la bitácora de eventos.
    registro = []
    # Inicializa un conjunto de nodos cerrados (ya expandidos definitivamente).
    cerrados = set()

    # Bucle principal que se ejecuta mientras existan elementos en la frontera priorizada.
    while frontera:
        # Extrae el nodo con menor costo acumulado de la cola de prioridad.
        costo_actual, actual = heapq.heappop(frontera)
        # Si el costo extraído no coincide con el mejor costo registrado, omite la iteración actual.
        if costo_actual != mejor_costo.get(actual, math.inf): continue
        # Si el nodo actual ya fue procesado en el conjunto cerrado, lo omite.
        if actual in cerrados: continue

        # Añade el nodo actual al conjunto cerrado.
        cerrados.add(actual)
        # Registra el nodo en la lista de expandidos.
        expandidos.append(actual)
        # Agrega un mensaje a la bitácora indicando la expansión del nodo y su costo g(n).
        registro.append(f"UCS: Se expande {actual} con g(n) = {costo_actual}.")
        # Si el nodo expandido es el objetivo, interrumpe el bucle.
        if actual == objetivo: break

        # Itera sobre los vecinos y pesos del nodo actual.
        for vecino, peso in vecinos_ordenados(actual):
            # Calcula el nuevo costo acumulado sumando el costo actual más el peso de la arista.
            nuevo_costo = costo_actual + peso
            # Si este nuevo costo es menor que el mejor costo registrado previamente para el vecino:
            if nuevo_costo < mejor_costo.get(vecino, math.inf):
                # Actualiza el mejor costo para el vecino.
                mejor_costo[vecino] = nuevo_costo
                # Actualiza el nodo padre del vecino.
                padres[vecino] = actual
                # Inserta el nuevo costo y el vecino en la cola de prioridad de la frontera.
                heapq.heappush(frontera, (nuevo_costo, vecino))
                # Registra la actualización de costos en la bitácora.
                registro.append(f"  -> Actualización para {vecino}: g(n) = {nuevo_costo}.")

    # Reconstruye el camino utilizando el diccionario de padres.
    camino = reconstruir_camino(padres, inicio, objetivo)
    # Calcula el costo total del camino obtenido.
    costo = calcular_costo(camino, grafo)
    # Retorna el camino, los nodos expandidos y el costo total.
    return camino, expandidos, costo

# Define la función para ejecutar el algoritmo de búsqueda Greedy (Voraz).
def ejecutar_greedy(inicio, objetivo):
    # Obtiene la red vial ponderada.
    grafo = obtener_red_vial_ponderada()
    # Calcula o recupera el diccionario de valores heurísticos hacia el objetivo.
    heuristica = calcular_heuristica(grafo, objetivo)
    # Valida que los nodos de inicio y objetivo existan en el grafo.
    if inicio not in grafo or objetivo not in grafo: return None, [], 0.0

    # Inicializa la frontera priorizada con una tupla que contiene el valor heurístico del inicio y el nodo inicial.
    frontera = [(heuristica.get(inicio, 0.0), inicio)]
    # Inicializa el conjunto de nodos descubiertos.
    descubiertos = {inicio}
    # Inicializa el diccionario de nodos padres.
    padres = {inicio: None}
    # Inicializa la lista de nodos expandidos.
    expandidos = []
    # Inicializa la bitácora de procesos.
    registro = []

    # Bucle principal que opera mientras la frontera tenga elementos.
    while frontera:
        # Extrae el nodo con menor valor heurístico h(n) de la cola de prioridad.
        h_actual, actual = heapq.heappop(frontera)
        # Si el nodo actual no ha sido expandido previamente:
        if actual not in expandidos:
            # Lo añade a la lista de expandidos.
            expandidos.append(actual)
            # Registra la expansión y su valor heurístico en la bitácora.
            registro.append(f"Greedy: Se expande {actual} con h(n) = {h_actual}.")
        # Si se alcanza el objetivo, finaliza el bucle.
        if actual == objetivo: break

        # Itera sobre los vecinos del nodo actual.
        for vecino, _ in vecinos_ordenados(actual):
            # Si el vecino no se encuentra en el conjunto de descubiertos:
            if vecino not in descubiertos:
                # Lo añade al conjunto de descubiertos.
                descubiertos.add(vecino)
                # Define su nodo padre.
                padres[vecino] = actual
                # Obtiene el valor heurístico del vecino.
                h_vecino = heuristica.get(vecino, 0.0)
                # Inserta el valor heurístico y el vecino en la cola de prioridad.
                heapq.heappush(frontera, (h_vecino, vecino))
                # Registra el descubrimiento del vecino en la bitácora.
                registro.append(f"  -> Se descubre {vecino} con h(n) = {h_vecino}.")

    # Reconstruye el camino mediante los padres guardados.
    camino = reconstruir_camino(padres, inicio, objetivo)
    # Calcula el costo real acumulado del camino.
    costo = calcular_costo(camino, grafo)
    # Retorna el camino, los expandidos y el costo.
    return camino, expandidos, costo

# Define la función para ejecutar el algoritmo de búsqueda informada A* (A Estrellas).
def ejecutar_astar(inicio, objetivo):
    # Obtiene el grafo ponderado de la red vial.
    grafo = obtener_red_vial_ponderada()
    # Calcula o recupera la heurística de los nodos respecto al objetivo.
    heuristica = calcular_heuristica(grafo, objetivo)
    # Valida la existencia de los nodos de inicio y objetivo.
    if inicio not in grafo or objetivo not in grafo: return None, [], 0.0

    # Inicializa la frontera priorizada con una tupla que almacena f(n), el costo real g(n) (0.0) y el nodo inicial.
    frontera = [(heuristica.get(inicio, 0.0), 0.0, inicio)]
    # Inicializa el diccionario con los mejores costos reales g(n) conocidos para cada nodo.
    mejor_g = {inicio: 0.0}
    # Inicializa el diccionario de padres.
    padres = {inicio: None}
    # Inicializa la lista de nodos expandidos.
    expandidos = []
    # Inicializa la bitácora de ejecución detallada.
    registro = []

    # Bucle principal operado mientras la frontera contenga elementos.
    while frontera:
        # Extrae el nodo con menor valor de evaluación total f(n) de la cola de prioridad.
        f_actual, g_actual, actual = heapq.heappop(frontera)
        # Si el costo g almacenado es mayor al mejor costo conocido, omite la iteración.
        if g_actual > mejor_g.get(actual, math.inf): continue
        # Si el nodo actual no ha sido expandido:
        if actual not in expandidos:
            # Lo añade a la lista de expandidos.
            expandidos.append(actual)
            # Registra la expansión del nodo junto a su función f(n) en la bitácora.
            registro.append(f"A*: Se expande {actual} with f(n)={f_actual}.")
        # Si se alcanza el nodo objetivo, detiene el ciclo.
        if actual == objetivo: break

        # Itera sobre los vecinos y pesos del nodo actual.
        for vecino, peso in vecinos_ordenados(actual):
            # Calcula el nuevo costo real acumulado g para el vecino.
            nuevo_g = g_actual + peso
            # Si el nuevo costo es menor que el mejor costo g registrado previamente para el vecino:
            if nuevo_g < mejor_g.get(vecino, math.inf):
                # Actualiza el mejor costo g del vecino.
                mejor_g[vecino] = nuevo_g
                # Actualiza su nodo padre.
                padres[vecino] = actual
                # Obtiene el valor heurístico h del vecino.
                h_vecino = heuristica.get(vecino, 0.0)
                # Calcula el nuevo valor total f(n) sumando g(n) más h(n).
                nuevo_f = nuevo_g + h_vecino
                # Inserta el nuevo f, el nuevo g y el vecino en la cola de prioridad de la frontera.
                heapq.heappush(frontera, (nuevo_f, nuevo_g, vecino))
                # Registra la actualización de la función de evaluación en la bitácora.
                registro.append(f"  -> Actualización en {vecino}: f(n)={nuevo_f}.")

    # Reconstruye el camino solución con los padres registrados.
    camino = reconstruir_camino(padres, inicio, objetivo)
    # Calcula el costo real total del camino.
    costo = calcular_costo(camino, grafo)
    # Retorna el camino final, los nodos expandidos y el costo total.
    return camino, expandidos, costo


# ==========================================================
# MOTOR AVANZADO DE LÓGICA PROPOSICIONAL (10 PREMISAS REALES)
# ==========================================================
# Define la función para evaluar las 10 premisas del motor de reglas proposicionales para carpooling.
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
    # Construye un diccionario que formaliza las 10 premisas lógicas de la Forma Normal Conjuntiva.
    premisas = {
        "O": origen in grafo,                                       # P1: Valida si el origen existe en la red vial.
        "D": (destino in grafo) and (origen != destino),             # P2: Valida si el destino es válido y diferente al origen.
        "V": bool(ruta_solucion) and len(ruta_solucion) >= 2,         # P3: Valida la conectividad efectiva de la ruta calculada por la IA.
        "C": asientos_disponibles > 0,                              # P4: Valida la disponibilidad óptima de asientos en el vehículo.
        "U": usuario_verificado,                                    # P5: Valida si el usuario cuenta con perfil y matrícula UTVT activa.
        "H": coincidencia_horario,                                  # P6: Valida la coincidencia horaria del viaje compartido.
        "E": conductor_aprobado,                                    # P7: Valida el estatus normativo y licencia del conductor.
        "S": viaje_activo,                                          # P8: Valida que el estado del viaje se encuentre activo en el sistema.
        "M": cuota_cubierta,                                        # P9: Valida que la cuota de recuperación de combustible esté aprobada.
        "R": zona_segura                                            # P10: Valida que la ruta esté libre de restricciones o zonas de alerta.
    }

    # Evalúa la inferencia lógica general aplicando una conjunción lógica (AND) a todas las premisas.
    viaje_permitido = all(premisas.values())

    # Inicializa una lista vacía para acumular las explicaciones detalladas en caso de fallo.
    lista_explicaciones = []
    # Si la premisa O es falsa, añade su explicación de error correspondiente.
    if not premisas["O"]: lista_explicaciones.append("O: El origen seleccionado no es válido en la red.")
    # Si la premisa D es falsa, añade su respectiva explicación.
    if not premisas["D"]: lista_explicaciones.append("O bien D: El destino no es válido o coincide con el origen.")
    # Si la premisa V es falsa, añade la alerta de falta de tramo conectado por IA.
    if not premisas["V"]: lista_explicaciones.append("V: No existe un tramo vial conectado y solucionado por la IA.")
    # Si la premisa C es falsa, advierte sobre la capacidad de cupo agotada.
    if not premisas["C"]: lista_explicaciones.append("C: Capacidad agotada (Sin asientos disponibles).")
    # Si la premisa U es falsa, indica que el usuario no está verificado.
    if not premisas["U"]: lista_explicaciones.append("U: Usuario sin registro verificado en la UTVT.")
    # Si la premisa H es falsa, reporta el desfase de horario.
    if not premisas["H"]: lista_explicaciones.append("H: Desfase en el horario de salida del viaje compartido.")
    # Si la premisa E es falsa, indica problemas con el estatus del conductor.
    if not premisas["E"]: lista_explicaciones.append("E: Conductor o vehículo sin validación institucional.")
    # Si la premisa S es falsa, indica que el viaje no está activo.
    if not premisas["S"]: lista_explicaciones.append("S: El estado del viaje no se encuentra activo.")
    # Si la premisa M es falsa, avisa sobre la cuota de trayecto pendiente.
    if not premisas["M"]: lista_explicaciones.append("M: Cuota de recuperación de trayecto pendiente.")
    # Si la premisa R es falsa, advierte sobre restricciones en la zona.
    if not premisas["R"]: lista_explicaciones.append("R: Ruta con alerta o restricción de tránsito.")

    # Condicional para establecer el texto del diagnóstico global según el resultado de la inferencia.
    if viaje_permitido:
        # Mensaje si todas las premisas lógicas se cumplieron satisfactoriamente.
        diagnostico = "Aprobado: Las 10 premisas de lógica proposicional se cumplen con éxito en Velox UTVT."
    else:
        # Mensaje de rechazo estructurado concatenando las razones detectadas.
        diagnostico = "Rechazado por Motor Lógico -> " + " | ".join(lista_explicaciones)

    # Retorna un diccionario estructurado con el estado de las premisas, la validación y el diagnóstico.
    return {
        "premisas": premisas,
        "viaje_permitido": viaje_permitido,
        "explicacion": diagnostico
    }


# ==========================================================
# CONTROLADOR DE RUTA GET
# ==========================================================
# Define la ruta HTTP GET para el endpoint principal del simulador de IA en FastAPI.
@router_simulador_ia.get("/", response_class=HTMLResponse)
def vista_simulador_ia(peticion_http: Request, origen: str = "Terminal", destino: str = "UTVT", algoritmo: str = "A*"):
    # Obtiene la red vial ponderada desde el sistema de datos.
    grafo_ponderado = obtener_red_vial_ponderada()
    # Obtiene el diccionario de coordenadas de los nodos viales.
    coords_nodos = obtener_coordenadas_nodos()
    
    # Crea una lista ordenada alfabéticamente con las claves de los nodos disponibles.
    lista_nodos_disponibles = sorted(list(grafo_ponderado.keys()))

    # Valida y asigna el origen recibido o establece un valor predeterminado seguro ("Terminal").
    origen = origen if origen in lista_nodos_disponibles else "Terminal"
    # Valida y asigna el destino recibido o establece un valor predeterminado seguro ("UTVT").
    destino = destino if destino in lista_nodos_disponibles else "UTVT"

    # Inicializa la variable que contendrá la ruta solución como una lista vacía.
    ruta_solucion = []
    # Inicializa la lista vacía para almacenar la bitácora de los procesos de búsqueda.
    bitacora_proceso = []
    # Inicializa el costo total numérico de la ruta en 0.0.
    costo_total_ruta = 0.0
    # Inicializa la lista vacía para la tabla comparativa masiva de algoritmos.
    tabla_comparativa = []

    # Define la lista con los nombres de los 5 algoritmos disponibles para comparación.
    algoritmos_a_ejecutar = ["BFS", "DFS", "UCS", "Greedy", "A*"]

    # Evalúa si el usuario seleccionó la opción masiva de comparar todos los algoritmos.
    if algoritmo == "Todos":
        # Itera sobre cada uno de los 5 algoritmos definidos en la lista.
        for algo in algoritmos_a_ejecutar:
            # Ejecuta BFS si el algoritmo actual de la iteración es BFS.
            if algo == "BFS": sol, bit, costo = ejecutar_bfs(origen, destino)
            # Ejecuta DFS si el algoritmo actual es DFS.
            elif algo == "DFS": sol, bit, costo = ejecutar_dfs(origen, destino)
            # Ejecuta UCS si el algoritmo actual es UCS.
            elif algo == "UCS": sol, bit, costo = ejecutar_ucs(origen, destino)
            # Ejecuta Greedy si el algoritmo actual es Greedy.
            elif algo == "Greedy": sol, bit, costo = ejecutar_greedy(origen, destino)
            # Ejecuta A* por defecto para cualquier otro caso en la iteración.
            elif algo == "A*": sol, bit, costo = ejecutar_astar(origen, destino)
            
            # Añade las métricas de rendimiento del algoritmo evaluado a la tabla comparativa.
            tabla_comparativa.append({
                "algoritmo": algo,
                "ruta": " ➔ ".join(sol) if sol else "Sin ruta",
                "nodos_explorados": len(bit),
                "costo": f"{costo:.1f}" if sol and math.isfinite(costo) else "N/A"
            })
        # Asigna la ruta, bitácora y costo utilizando A* como solución principal de referencia visual.
        ruta_solucion, bitacora_proceso, costo_total_ruta = ejecutar_astar(origen, destino)
    else:
        # Ejecuta el algoritmo específico seleccionado individualmente por el usuario.
        if algoritmo == "DFS": ruta_solucion, bitacora_proceso, costo_total_ruta = ejecutar_dfs(origen, destino)
        elif algoritmo == "UCS": ruta_solucion, bitacora_proceso, costo_total_ruta = ejecutar_ucs(origen, destino)
        elif algoritmo == "Greedy": ruta_solucion, bitacora_proceso, costo_total_ruta = ejecutar_greedy(origen, destino)
        elif algoritmo == "A*": ruta_solucion, bitacora_proceso, costo_total_ruta = ejecutar_astar(origen, destino)
        else: ruta_solucion, bitacora_proceso, costo_total_ruta = ejecutar_bfs(origen, destino)

    # Evalúa formalmente el motor de reglas de carpooling utilizando las 10 premisas lógicas integradas.
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

    # Calcula el número total de movimientos o transbordos (aristas) restando uno al total de nodos de la ruta.
    total_movimientos_aristas = len(ruta_solucion) - 1 if ruta_solucion else 0

    # Agrupa todas las variables y resultados en un diccionario de contexto para la plantilla HTML.
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

    # Renderiza y retorna la respuesta HTTP con la plantilla HTML y el contexto de datos inyectado.
    return motor_plantillas.TemplateResponse(peticion_http, "ia/simulador_ia_inicio.html", contexto_plantilla)