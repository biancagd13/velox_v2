# ============================================================
# VELOX UTVT - BUSQUEDA DE CONEXIONES ENTRE ZONAS (BFS)
# Conectado a MongoDB (db.ruta + db.viaje) con respaldo de grafo base
# ============================================================

from collections import deque
import os
import sys
from typing import Any

RAIZ_PROYECTO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if RAIZ_PROYECTO not in sys.path:
    sys.path.append(RAIZ_PROYECTO)

try:
    from database import db
except ImportError:
    db = None


GRAFO_BASE_VELOX: dict[str, list[str]] = {
    "Toluca": ["Metepec Centro", "San Mateo Atenco", "Lerma", "Utvt"],
    "Metepec Centro": ["Plaza Las Americas", "Lerma Centro", "Utvt"],
    "Plaza Las Americas": ["Tecnologico", "Utvt"],
    "Lerma": ["Santa Maria Atarasquillo", "Utvt"],
    "Lerma Centro": ["Santa Maria Atarasquillo", "Utvt"],
    "San Mateo Atenco": ["Lerma", "Utvt"],
    "Tecnologico": ["Utvt"],
    "Santa Maria Atarasquillo": ["Utvt"],
    "Utvt": [],
}


def normalizar_zona(nombre: str | None) -> str:
    """Limpia y estandariza el nombre de una zona o punto."""
    if not nombre:
        return ""
    return " ".join(str(nombre).strip().split()).title()


def construir_grafo_desde_bd(grafo_base: dict[str, list[str]] | None = None) -> dict[str, list[str]]:
    """
    Construye un grafo dirigido origen -> destino usando db.ruta y db.viaje
    en MongoDB y conservando el grafo base como respaldo.
    """
    grafo: dict[str, list[str]] = {
        normalizar_zona(origen): [
            normalizar_zona(dest) for dest in destinos if normalizar_zona(dest)
        ]
        for origen, destinos in (grafo_base or GRAFO_BASE_VELOX).items()
        if normalizar_zona(origen)
    }

    if db is None:
        return grafo

    def _agregar_arista(origen_raw: Any, destino_raw: Any) -> None:
        origen = normalizar_zona(origen_raw)
        destino = normalizar_zona(destino_raw)
        if not origen or not destino:
            return
        grafo.setdefault(origen, [])
        grafo.setdefault(destino, [])
        if destino not in grafo[origen]:
            grafo[origen].append(destino)

    try:
        rutas_por_id: dict[str, dict[str, Any]] = {}
        for r in db["ruta"].find({}, {"_id": 0}):
            _agregar_arista(r.get("origen"), r.get("destino"))
            if r.get("id_ruta") is not None:
                rutas_por_id[str(r["id_ruta"]).strip()] = r

        for v in db["viaje"].find({}, {"_id": 0}):
            ruta_ref = rutas_por_id.get(str(v.get("id_ruta") or "").strip(), {})
            _agregar_arista(
                v.get("origen") or ruta_ref.get("origen"),
                v.get("destino") or ruta_ref.get("destino"),
            )
    except Exception as error:
        print(f"[Aviso BFS] No se pudieron consultar rutas en BD: {error}")

    return grafo


def buscar_ruta_bfs(
    grafo: dict[str, list[str]],
    inicio: str,
    destino: str,
) -> list[str] | None:
    """Busca una ruta entre 'inicio' y 'destino' mediante BFS."""
    nodo_inicio = normalizar_zona(inicio)
    nodo_destino = normalizar_zona(destino)

    if not nodo_inicio or not nodo_destino:
        return None

    if nodo_inicio == nodo_destino:
        return [nodo_inicio]

    if nodo_inicio not in grafo:
        return None

    cola: deque[list[str]] = deque([[nodo_inicio]])
    visitados: set[str] = {nodo_inicio}

    while cola:
        ruta = cola.popleft()
        actual = ruta[-1]

        for vecino in grafo.get(actual, []):
            if vecino == nodo_destino:
                return ruta + [vecino]

            if vecino not in visitados:
                visitados.add(vecino)
                cola.append(ruta + [vecino])

    return None


def obtener_ruta_velox(inicio: str, destino: str) -> dict[str, Any]:
    """Obtiene el grafo actualizado desde BD (con respaldo) y ejecuta BFS."""
    grafo = construir_grafo_desde_bd()
    ruta = buscar_ruta_bfs(grafo, inicio, destino)

    if ruta is None:
        return {
            "encontrada": False,
            "inicio": normalizar_zona(inicio),
            "destino": normalizar_zona(destino),
            "ruta": [],
            "conexiones": 0,
            "mensaje": "No se encontro conexion directa o intermedia entre las zonas.",
        }

    return {
        "encontrada": True,
        "inicio": normalizar_zona(inicio),
        "destino": normalizar_zona(destino),
        "ruta": ruta,
        "conexiones": max(0, len(ruta) - 1),
        "mensaje": "Ruta encontrada mediante BFS.",
    }