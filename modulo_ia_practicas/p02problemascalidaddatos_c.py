# ============================================================
# PRACTICA 2 ADAPTADA A VELOX UTVT - PARTE C
# DETECCION DE ERRORES EN LOS DATOS
# ============================================================

# ¿Por qué es importante limpiar los datos?
# Imagina que alguien registra por error:
# - Una puntualidad o confianza de 110%.
# - Una calificación de conductor vacía.
# - El mismo usuario registrado dos veces.
# - Un valor de viajes compartidos como texto en lugar de número.
# Si utilizamos esos datos sin revisarlos, los resultados del análisis y del modelo podrían ser incorrectos.

# Para practicar, introduciremos algunos errores deliberadamente.

import pandas as pd

# Crear un conjunto pequeño con errores intencionales adaptado a Velox UTVT.
datos_prueba_velox = {
    "usuario": [
        "Alexa", "Brandon", "Josh", "Bianca", "Bryan"
    ],

    "viajes_compartidos": [
        8, 3, None, 3, 5
    ],

    "puntualidad": [
        95, 60, 98, 60, 110
    ],

    "calificacion_conductor": [
        100, 58, 96, 58, None
    ]
}

df = pd.DataFrame(datos_prueba_velox)

print("=== DATOS DE VELOX CON PROBLEMAS INTENCIONALES ===")
print(df)

# ------------------------------------------------------------
# 1. BUSCAR VALORES FALTANTES
# ------------------------------------------------------------
# isnull() identifica los valores ausentes (NaN / None).
# sum() cuenta cuántos hay en cada columna.

print("\n=== VALORES FALTANTES POR COLUMNA EN VELOX ===")
print(df.isnull().sum())

# ------------------------------------------------------------
# 2. CONTAR FILAS CON ALGUN DATO FALTANTE
# ------------------------------------------------------------
filas_incompletas = df.isnull().any(axis=1).sum()

print("\nFilas con al menos un valor faltante en la red:")
print(filas_incompletas)

# ------------------------------------------------------------
# 3. DETECTAR REGISTROS DUPLICADOS
# ------------------------------------------------------------
# duplicated() detecta filas completamente repetidas.
# No identifica necesariamente usuarios repetidos
# cuando sus otros datos o métricas son diferentes.

print("\n=== FILAS DUPLICADAS EN EL REGISTRO ===")
print(df[df.duplicated()])

# ------------------------------------------------------------
# 4. REVISAR LOS RANGOS DE LAS VARIABLES
# ------------------------------------------------------------
# En este ejemplo de Velox, la puntualidad y la calificación
# deberían estar estrictamente entre 0 y 100 por ciento o puntos.

puntualidad_invalida = (
    (df["puntualidad"] < 0) |
    (df["puntualidad"] > 100)
)

print("\n=== PUNTUALIDADES FUERA DE RANGO ===")
print(df[puntualidad_invalida])

calificacion_invalida = (
    (df["calificacion_conductor"] < 0) |
    (df["calificacion_conductor"] > 100)
)

print("\n=== CALIFICACIONES DE CONDUCTOR FUERA DE RANGO ===")
print(df[calificacion_invalida])