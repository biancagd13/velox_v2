# ============================================================
# PRACTICA 2 ADAPTADA A VELOX UTVT - PARTE A
# CREACION DE UN CONJUNTO DE DATOS DE LA COMUNIDAD
# ============================================================

# pandas es una biblioteca de Python para trabajar con tablas.

# Al ejecutar el código, se habrá creado una tabla y la habrán guardado en un archivo CSV.

# El archivo CSV es importante porque permite almacenar los datos de forma independiente del programa. 
# Más adelante podremos cargar ese archivo para entrenar distintos modelos sin tener que escribir 
# nuevamente toda la información.

# Actividad rápida: agregar mas usuarios con datos inventados y comprobar cómo cambian las 
# dimensiones de la tabla.

import os
import pandas as pd

# ------------------------------------------------------------
# PASO 1. CREAR LOS DATOS
# ------------------------------------------------------------
# Cada diccionario representa una columna.
# Las listas contienen los valores correspondientes a cada fila.

datos = {
    "usuario": [
        "Alexa", "Brandon", "Josh", "Bryan", "Bianca",
        "Pepe", "Eren", "Nanami"
    ],

    "viajes_compartidos": [
        8, 3, 10, 5, 6, 2, 7, 4
    ],

    "puntualidad": [
        95, 60, 98, 75, 85, 55, 90, 70
    ],

    "calificacion_conductor": [
        100, 50, 100, 80, 90, 40, 95, 65
    ],

    "nivel_confianza": [
        92, 58, 96, 72, 81, 50, 88, 64
    ]
}

# ------------------------------------------------------------
# PASO 2. CONVERTIR LOS DATOS EN UN DATAFRAME
# ------------------------------------------------------------
# Un DataFrame es una estructura tabular de pandas.
# Podemos imaginarlo como una hoja de cálculo.

df = pd.DataFrame(datos)

# ------------------------------------------------------------
# PASO 3. MOSTRAR LA TABLA
# ------------------------------------------------------------
print("=== CONJUNTO DE DATOS ORIGINAL DE VELOX UTVT ===")
print(df)

# ------------------------------------------------------------
# PASO 4. CONSULTAR EL TAMANO DE LA TABLA
# ------------------------------------------------------------
# shape devuelve una tupla:
# (numero de filas, numero de columnas)

print("\nDimensiones de la tabla:")
print(df.shape)

print("Numero de usuarios:", df.shape[0])
print("Numero de variables:", df.shape[1])

# ------------------------------------------------------------
# PASO 5. GUARDAR LOS DATOS EN UN ARCHIVO CSV (DE FORMA SEGURA)
# ------------------------------------------------------------
# Obtener la ruta absoluta de la carpeta donde se encuentra este script
carpeta_actual = os.path.dirname(os.path.abspath(__file__))
ruta_salida = os.path.join(carpeta_actual, "usuarios_velox.csv")

# index=False evita agregar una columna adicional con
# los indices internos de pandas.
df.to_csv(
    ruta_salida,
    index=False,
    encoding="utf-8"
)

print(f"\nArchivo usuarios_velox.csv creado correctamente en: {ruta_salida}")