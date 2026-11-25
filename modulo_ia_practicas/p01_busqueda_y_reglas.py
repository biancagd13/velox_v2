# ============================================================
# PRACTICA 1 ADAPTADA A VELOX UTVT: BUSQUEDA FRENTE A REGLAS
# PARTE A: ENCONTRAR UNA RUTA DE CARPOOLING MEDIANTE BFS
# ============================================================

# Objetivo de aprendizaje:
# Que el estudiante identifique las diferencias entre resolver 
# un problema de movilidad mediante reglas y algoritmos de búsqueda, 
# y resolverlo mediante un modelo que aprende de datos históricos.

# Planteamiento del problema:
# Imagina que la plataforma de carpooling Velox UTVT necesita 
# calcular una ruta de conexión eficiente para la comunidad universitaria 
# desde los puntos de origen (como la Terminal o municipios cercanos) 
# hasta el campus de la UTVT utilizando un grafo vial.

# El sistema necesita realizar tareas de:
# 1. Búsqueda de rutas viales óptimas mediante grafos y BFS.
# 2. Validación de reglas lógicas para la asignación de viajes compartidos.

# Al terminar, serás capaz de:
# - Identificar entradas, estados, objetivos y resultados en Velox UTVT.
# - Distinguir un algoritmo tradicional de búsqueda de un modelo de aprendizaje automático.
# - Implementar soluciones basadas en la red de transporte de la universidad.

from collections import deque

# ------------------------------------------------------------
# 1. REPRESENTAR LA RED VIAL DE VELOX UTVT COMO UN GRAFO
# ------------------------------------------------------------
# Cada ubicación o municipio es un nodo.
# Cada conexión representa un tramo vial directo disponible en la plataforma.
#
# Por ejemplo:
# Terminal -> San Mateo significa que existe un trayecto
# directo registrado en el sistema de carpooling.

mapa_velox = {
    "Terminal": ["San Mateo", "Toluca (Centro)"],
    "San Mateo": ["UTVT"],
    "Toluca (Centro)": ["Metepec"],
    "Metepec": ["Lerma"],
    "Lerma": ["UTVT"],
    "UTVT": []
}


# ------------------------------------------------------------
# 2. DEFINIR EL ALGORITMO DE BUSQUEDA EN ANCHURA (BFS)
# ------------------------------------------------------------
def buscar_ruta_velox(grafo, inicio, objetivo):

    # La cola almacena las rutas pendientes de explorar en la red de Velox.
    # BFS utiliza una cola FIFO: primero entra, primero sale.
    cola = deque([[inicio]])

    # Registrar los nodos visitados evita repetir exploraciones en el mapa.
    visitados = {inicio}

    # Continuar mientras existan rutas por explorar en el sistema.
    while cola:

        # Extraer la ruta más antigua de la cola.
        ruta = cola.popleft()

        # El último nodo de la ruta es nuestra posición actual en el trayecto.
        nodo_actual = ruta[-1]

        # Comprobar si llegamos al campus destino (UTVT).
        if nodo_actual == objetivo:
            return ruta

        # Explorar los vecinos (conexiones viales) del nodo actual.
        for vecino in grafo.get(nodo_actual, []):

            # Solo incorporar nodos que todavía no visitamos.
            if vecino not in visitados:

                visitados.add(vecino)

                # Crear una nueva ruta sin modificar la anterior.
                nueva_ruta = ruta + [vecino]

                # Agregar la nueva ruta a la cola de pendientes.
                cola.append(nueva_ruta)

    # Si agotamos la cola, no se encontró una ruta conectada en Velox.
    return None


# ------------------------------------------------------------
# 3. EJECUTAR EL ALGORITMO EN VELOX UTVT
# ------------------------------------------------------------
inicio = "Terminal"
objetivo = "UTVT"

ruta_optima = buscar_ruta_velox(mapa_velox, inicio, objetivo)

print("=== RESULTADO DE LA BUSQUEDA EN VELOX UTVT ===")

if ruta_optima is not None:
    print("Ruta de carpooling encontrada:")
    print(" -> ".join(ruta_optima))
    print("Número de tramos (conexiones):", len(ruta_optima) - 1)
else:
    print("No existe una ruta vial conectada hacia la UTVT.")