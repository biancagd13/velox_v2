# ============================================================
# PRACTICA 2 ADAPTADA A VELOX UTVT - PARTE E
# PREPARAR LAS VARIABLES DE ENTRADA Y EL OBJETIVO
# ============================================================

# Supongamos que queremos entrenar un modelo para estimar el nivel de confianza de un usuario en Velox.

# Definimos:
# - Variables de entrada (X): viajes compartidos, puntualidad y calificación de conductor.
# - Variable objetivo (y): nivel de confianza.
# En notación habitual de aprendizaje automático:

# X = [viajes compartidos, puntualidad, calificación de conductor]

# y = [nivel de confianza]

# Todavía no estamos entrenando un modelo. 
# Únicamente estamos preparando la estructura de los datos que necesitará.

import os
import pandas as pd

# Cargar el conjunto de datos original de manera segura usando os.path.
carpeta_actual = os.path.dirname(os.path.abspath(__file__))
ruta_csv = os.path.join(carpeta_actual, "usuarios_velox.csv")

df = pd.read_csv(ruta_csv)

# ------------------------------------------------------------
# 1. SELECCIONAR LAS VARIABLES DE ENTRADA
# ------------------------------------------------------------
# X contiene las características que utilizaremos
# para explicar o estimar el nivel de confianza del usuario.

X = df[
    [
        "viajes_compartidos",
        "puntualidad",
        "calificacion_conductor"
    ]
]

# ------------------------------------------------------------
# 2. SELECCIONAR LA VARIABLE OBJETIVO
# ------------------------------------------------------------
# y contiene el resultado que queremos estimar.

y = df["nivel_confianza"]

# ------------------------------------------------------------
# 3. MOSTRAR LOS DATOS
# ------------------------------------------------------------
print("=== VARIABLES DE ENTRADA: X (VELOX) ===")
print(X.head())

print("\n=== VARIABLE OBJETIVO: y (VELOX) ===")
print(y.head())

print("\nDimensiones de X:", X.shape)
print("Dimensiones de y:", y.shape)

# ------------------------------------------------------------
# 4. COMPROBAR LA CORRESPONDENCIA ENTRE X E y
# ------------------------------------------------------------
# Cada fila de X debe corresponder al mismo usuario
# que el valor ubicado en la misma posición de y.

assert len(X) == len(y), "X e y tienen diferente cantidad de filas"

print("\nLas entradas y el objetivo de Velox están alineados.")


# Un modelo supervisado utiliza ejemplos en los que conocemos 
# tanto las entradas como el resultado esperado. Durante el entrenamiento,
# el modelo busca relaciones entre ambos.

# Por ejemplo, el modelo recibirá información sobre los viajes compartidos, 
# la puntualidad y la calificación, y utilizará los niveles de confianza 
# conocidos para ajustar sus parámetros.

# Sin embargo, debemos tener cuidado: los datos de esta práctica son de prueba 
# y muy pequeños. Sirven para aprender el procedimiento, 
# no para crear un predictor operativo de producción definitivo.