# ============================================================
# PROYECTO: VELOX UTVT
# PRACTICA 16 (ETAPA 7): CONSULTAS INTELIGENTES Y ANALISIS ESTADISTICO
#
# Archivo:
# practica16_consultas_inteligentes.py
#
# Objetivo:
# Consultar los registros de viajes almacenados en MongoDB Atlas
# y generar estadísticas mediante el framework de agregación para
# identificar patrones operativos, perfiles de trayectos y riesgos.
# ============================================================

import os
import sys
import pandas as pd
from pathlib import Path
from pymongo.errors import PyMongoError

# Permitir importar 'database.py' desde la raíz del proyecto Velox UTVT
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from database import db

# ============================================================
# 1. CONFIGURACION DE DIRECTORIOS
# ============================================================

RUTA_ACTUAL = Path(__file__).resolve().parent
RUTA_RESULTADOS = RUTA_ACTUAL / "resultados_p16_velox"
RUTA_RESULTADOS.mkdir(parents=True, exist_ok=True)


# ============================================================
# 2. CONSULTAR TODOS LOS VIAJES REGISTRADOS (find)
# ============================================================

def consultar_todos(coleccion):
    """
    Recupera todos los viajes registrados en la colección 'viaje'
    de Velox UTVT ordenados de forma descendente.
    """
    print("\n1. CONSULTA GENERAL DE VIAJES (VELOX UTVT)")
    print("-" * 65)

    cursor = coleccion.find(
        {},
        {
            "_id": 1,
            "origen": 1,
            "destino": 1,
            "costo": 1,
            "asientos_disponibles": 1,
            "estado": 1,
            "fecha": 1
        }
    ).sort("_id", -1)

    registros = list(cursor)
    print(f"Total de viajes encontrados en MongoDB: {len(registros)}")

    for numero, reg in enumerate(registros[:10], start=1):
        print(f"\nVIAJE #{numero} | ID: {reg.get('_id')}")
        print(f"  Ruta: {reg.get('origen', 'N/A')} -> {reg.get('destino', 'UTVT')}")
        print(f"  Costo: ${float(reg.get('costo', 0)):.2f} MXN")
        print(f"  Asientos disponibles: {reg.get('asientos_disponibles', 0)}")
        print(f"  Estado operativo: {reg.get('estado', 'desconocido')}")

    if len(registros) > 10:
        print(f"\n... (Mostrando 10 de {len(registros)} viajes totales)")

    return registros


# ============================================================
# 3. IDENTIFICAR VIAJES CON RIESGO OPERATIVO ($match, $project, $sort)
# ============================================================

def consultar_riesgos(coleccion):
    """
    Filtra viajes con riesgo operativo (estado cancelado/pendiente
    o tarifa elevada > $35 MXN con baja disponibilidad).
    """
    print("\n2. VIAJES CON RIESGO OPERATIVO O BAJA VIABILIDAD")
    print("-" * 65)

    pipeline = [
        # Etapa 1: filtrar viajes no exitosos o con tarifa alta
        {
            "$match": {
                "$or": [
                    {"estado": {"$in": ["cancelado", "pendiente"]}},
                    {"costo": {"$gt": 35}}
                ]
            }
        },
        # Etapa 2: proyectar campos relevantes y calcular índice de viabilidad
        {
            "$project": {
                "_id": 0,
                "origen": {"$ifNull": ["$origen", "No especificado"]},
                "destino": {"$ifNull": ["$destino", "UTVT"]},
                "costo": {"$toDouble": {"$ifNull": ["$costo", 0]}},
                "asientos_disponibles": {"$toInt": {"$ifNull": ["$asientos_disponibles", 0]}},
                "estado": {"$ifNull": ["$estado", "pendiente"]},
                "motivo_riesgo": {
                    "$cond": [
                        {"$in": ["$estado", ["cancelado", "pendiente"]]},
                        "Estado no consolidado (Cancelado/Pendiente)",
                        "Tarifa superior al promedio estudiantil (>$35)"
                    ]
                }
            }
        },
        # Etapa 3: ordenar de mayor a menor costo
        {
            "$sort": {"costo": -1}
        }
    ]

    resultados = list(coleccion.aggregate(pipeline))
    print(f"Viajes detectados con riesgo operativo: {len(resultados)}")

    for res in resultados[:10]:
        print(
            f"- {res['origen']} -> {res['destino']} | "
            f"Costo: ${res['costo']:.2f} | Asientos: {res['asientos_disponibles']} | "
            f"Estado: {res['estado']} ({res['motivo_riesgo']})"
        )

    return resultados


# ============================================================
# 4. AGRUPAR VIAJES POR ESTADO / PERFIL ($group)
# ============================================================

def analizar_perfiles_viaje(coleccion):
    """
    Agrupa los viajes por su estado operativo y calcula promedios
    de costo y asientos disponibles.
    """
    print("\n3. DISTRIBUCION POR ESTADO OPERATIVO DE VIAJES")
    print("-" * 65)

    pipeline = [
        {
            "$group": {
                "_id": {"$ifNull": ["$estado", "sin_estado"]},
                "total_viajes": {"$sum": 1},
                "promedio_costo": {"$avg": {"$toDouble": {"$ifNull": ["$costo", 0]}}},
                "promedio_asientos": {"$avg": {"$toInt": {"$ifNull": ["$asientos_disponibles", 0]}}},
                "costo_minimo": {"$min": {"$toDouble": {"$ifNull": ["$costo", 0]}}},
                "costo_maximo": {"$max": {"$toDouble": {"$ifNull": ["$costo", 0]}}}
            }
        },
        {
            "$sort": {"total_viajes": -1}
        }
    ]

    resultados = list(coleccion.aggregate(pipeline))

    for grupo in resultados:
        print(f"\nEstado / Perfil: {grupo.get('_id')}")
        print(f"  Cantidad de viajes: {grupo.get('total_viajes')}")
        print(f"  Promedio de costo: ${round(grupo.get('promedio_costo') or 0, 2)} MXN")
        print(f"  Promedio de asientos: {round(grupo.get('promedio_asientos') or 0, 2)}")
        print(f"  Rango de tarifas: ${grupo.get('costo_minimo', 0):.2f} - ${grupo.get('costo_maximo', 0):.2f}")

    return resultados


# ============================================================
# 5. ESTADISTICAS GENERALES DE LA PLATAFORMA ($group + $cond)
# ============================================================

def generar_estadisticas_generales(coleccion):
    """
    Calcula indicadores globales de toda la colección 'viaje'
    contabilizando viajes exitosos vs. no exitosos.
    """
    print("\n4. ESTADISTICAS GENERALES DE VELOX UTVT")
    print("-" * 65)

    pipeline = [
        {
            "$group": {
                "_id": None,
                "total_viajes": {"$sum": 1},
                "promedio_costo_global": {"$avg": {"$toDouble": {"$ifNull": ["$costo", 0]}}},
                "promedio_asientos_global": {"$avg": {"$toInt": {"$ifNull": ["$asientos_disponibles", 0]}}},
                "total_asientos_ofertados": {"$sum": {"$toInt": {"$ifNull": ["$asientos_disponibles", 0]}}},
                "total_viajes_exitosos": {
                    "$sum": {
                        "$cond": [
                            {"$in": ["$estado", ["activo", "en_curso", "finalizado"]]},
                            1,
                            0
                        ]
                    }
                },
                "total_viajes_riesgo": {
                    "$sum": {
                        "$cond": [
                            {"$in": ["$estado", ["cancelado", "pendiente"]]},
                            1,
                            0
                        ]
                    }
                }
            }
        }
    ]

    resultados = list(coleccion.aggregate(pipeline))

    if not resultados:
        print("No existen registros para analizar.")
        return None

    stats = resultados[0]
    total = stats.get("total_viajes", 0)
    exitosos = stats.get("total_viajes_exitosos", 0)
    tasa_exito = (exitosos / total * 100) if total > 0 else 0

    print(f"Total de viajes analizados: {total}")
    print(f"Promedio de costo global: ${round(stats.get('promedio_costo_global') or 0, 2)} MXN")
    print(f"Promedio de asientos por viaje: {round(stats.get('promedio_asientos_global') or 0, 2)}")
    print(f"Total de asientos ofertados: {stats.get('total_asientos_ofertados')}")
    print(f"Viajes exitosos (activo/en_curso/finalizado): {exitosos} ({tasa_exito:.2f}%)")
    print(f"Viajes en riesgo (cancelado/pendiente): {stats.get('total_viajes_riesgo')}")

    return stats


# ============================================================
# 6. TOP 5 VIAJES MAS ECONOMICOS Y DISPONIBLES ($sort + $limit)
# ============================================================

def consultar_mejores_viajes(coleccion):
    """
    Obtiene los 5 viajes activos/viables más económicos y con mayor
    disponibilidad de asientos para la comunidad universitaria.
    """
    print("\n5. TOP 5 VIAJES MAS ECONOMICOS Y VIABLES")
    print("-" * 65)

    pipeline = [
        {
            "$match": {
                "estado": {"$in": ["activo", "en_curso", "finalizado"]},
                "asientos_disponibles": {"$gte": 1}
            }
        },
        {
            "$project": {
                "_id": 0,
                "origen": {"$ifNull": ["$origen", "N/A"]},
                "destino": {"$ifNull": ["$destino", "UTVT"]},
                "costo": {"$toDouble": {"$ifNull": ["$costo", 0]}},
                "asientos_disponibles": {"$toInt": {"$ifNull": ["$asientos_disponibles", 0]}},
                "estado": 1
            }
        },
        {
            "$sort": {
                "costo": 1,
                "asientos_disponibles": -1
            }
        },
        {
            "$limit": 5
        }
    ]

    resultados = list(coleccion.aggregate(pipeline))

    for pos, viaje in enumerate(resultados, start=1):
        print(
            f"POSICION {pos} | {viaje['origen']} -> {viaje['destino']} | "
            f"Costo: ${viaje['costo']:.2f} MXN | "
            f"Asientos: {viaje['asientos_disponibles']} | Estado: {viaje['estado']}"
        )

    return resultados


# ============================================================
# 7. EXPORTAR REPORTES A ARCHIVOS CSV
# ============================================================

def exportar_reportes(registros, riesgos, perfiles, estadisticas):
    """
    Exporta los resultados de las consultas a 4 archivos CSV auditables.
    """
    print("\n6. EXPORTACION DE REPORTES CSV")
    print("-" * 65)

    # 1. Historial de viajes
    if registros:
        filas = []
        for reg in registros:
            estado = reg.get("estado", "")
            filas.append({
                "id_viaje": str(reg.get("_id")),
                "origen": reg.get("origen", ""),
                "destino": reg.get("destino", "UTVT"),
                "costo": float(reg.get("costo", 0)),
                "asientos_disponibles": int(reg.get("asientos_disponibles", 0)),
                "estado": estado,
                "viaje_exitoso": 1 if estado in ["activo", "en_curso", "finalizado"] else 0
            })
        ruta_hist = RUTA_RESULTADOS / "velox_analisis_historico.csv"
        pd.DataFrame(filas).to_csv(ruta_hist, index=False, encoding="utf-8-sig")
        print(f"- Reporte histórico guardado: {ruta_hist}")

    # 2. Viajes con riesgo
    if riesgos:
        ruta_riesgos = RUTA_RESULTADOS / "velox_analisis_riesgos.csv"
        pd.DataFrame(riesgos).to_csv(ruta_riesgos, index=False, encoding="utf-8-sig")
        print(f"- Reporte de riesgos guardado: {ruta_riesgos}")

    # 3. Estadísticas por estado/perfil
    if perfiles:
        df_perfiles = pd.DataFrame(perfiles).rename(columns={"_id": "estado_operativo"})
        ruta_perfiles = RUTA_RESULTADOS / "velox_estadisticas_estados.csv"
        df_perfiles.to_csv(ruta_perfiles, index=False, encoding="utf-8-sig")
        print(f"- Reporte por estados guardado: {ruta_perfiles}")

    # 4. Estadísticas generales
    if estadisticas:
        df_stats = pd.DataFrame([estadisticas]).drop(columns=["_id"], errors="ignore")
        ruta_stats = RUTA_RESULTADOS / "velox_estadisticas_generales.csv"
        df_stats.to_csv(ruta_stats, index=False, encoding="utf-8-sig")
        print(f"- Reporte general guardado: {ruta_stats}")


# ============================================================
# 8. FUNCION PRINCIPAL
# ============================================================

def main():
    print("=" * 65)
    print("VELOX UTVT")
    print("PRACTICA 16 - CONSULTAS INTELIGENTES Y ANALISIS ESTADISTICO")
    print("=" * 65)

    try:
        coleccion_viajes = db.viaje

        # Paso 1: consulta general con find()
        registros = consultar_todos(coleccion_viajes)

        if not registros:
            print("\nNo existen viajes registrados en la colección 'viaje' de Velox UTVT.")
            return

        # Paso 2: viajes con riesgo
        riesgos = consultar_riesgos(coleccion_viajes)

        # Paso 3: agrupación por estado/perfil
        perfiles = analizar_perfiles_viaje(coleccion_viajes)

        # Paso 4: estadísticas generales
        estadisticas = generar_estadisticas_generales(coleccion_viajes)

        # Paso 5: top 5 mejores opciones
        consultar_mejores_viajes(coleccion_viajes)

        # Paso 6: exportar CSVs
        exportar_reportes(registros, riesgos, perfiles, estadisticas)

        print("\n" + "=" * 65)
        print("PRACTICA 16 (VELOX UTVT) FINALIZADA CORRECTAMENTE")
        print("=" * 65)

    except PyMongoError as error:
        print(f"\nERROR DE MONGODB: {error}")
    except Exception as error:
        print(f"\nERROR INESPERADO: {error}")


if __name__ == "__main__":
    main()