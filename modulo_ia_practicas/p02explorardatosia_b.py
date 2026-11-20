# ============================================================
# PRACTICA 2 ADAPTADA A VELOX UTVT - PARTE B
# EXPLORACION DEL CONJUNTO DE DATOS
# ============================================================

# No basta con tener datos. Antes de utilizarlos, 
# necesitamos conocer su estructura, sus tipos y sus características estadísticas.
import os
import pandas as pd

# Leer el archivo CSV creado en la parte anterior (asegúrate de que esté en la misma carpeta o ruta correcta).
carpeta_actual = os.path.dirname(os.path.abspath(__file__))
ruta_csv = os.path.join(carpeta_actual, "usuarios_velox.csv")

df = pd.read_csv(ruta_csv)
# ------------------------------------------------------------
# 1. MOSTRAR LOS PRIMEROS REGISTROS
# ------------------------------------------------------------
# head() muestra las primeras cinco filas por defecto.

print("=== PRIMEROS REGISTROS DE VELOX ===")
print(df.head())

# ------------------------------------------------------------
# 2. MOSTRAR LOS ULTIMOS REGISTROS
# ------------------------------------------------------------
print("\n=== ULTIMOS REGISTROS DE VELOX ===")
print(df.tail())

# ------------------------------------------------------------
# 3. CONOCER LOS TIPOS DE DATOS
# ------------------------------------------------------------
# Los nombres de usuarios son texto.
# Los viajes, puntualidad, calificaciones y confianza son numéricos.

print("\n=== TIPOS DE DATOS EN LA PLATAFORMA ===")
print(df.dtypes)

# ------------------------------------------------------------
# 4. OBTENER INFORMACION GENERAL
# ------------------------------------------------------------
# info() muestra las columnas, los valores no nulos
# y los tipos de datos.

print("\n=== INFORMACION GENERAL DE LA RED ===")
df.info()

# ------------------------------------------------------------
# 5. CALCULAR ESTADISTICAS DESCRIPTIVAS
# ------------------------------------------------------------
# describe() calcula estadísticas para las columnas numéricas.
# Incluye media, desviación estándar, mínimo, máximo
# y algunos percentiles.

print("\n=== ESTADISTICAS DESCRIPTIVAS DE VIAJES Y USUARIOS ===")
print(df.describe())

# ------------------------------------------------------------
# 6. CONSULTAR UNA COLUMNA ESPECIFICA
# ------------------------------------------------------------
print("\n=== VIAJES COMPARTIDOS Y USUARIOS ===")
print(df[["viajes_compartidos", "usuario"]])

# ------------------------------------------------------------
# 7. CALCULAR EL PROMEDIO DE CONFIANZA
# ------------------------------------------------------------
promedio_confianza = df["nivel_confianza"].mean()

print("\nPromedio de nivel de confianza en Velox:", round(promedio_confianza, 2))

# Función      ¿Para qué sirve?
# head()       Muestra las primeras filas.
# tail()       Muestra las últimas filas.
# shape        Devuelve filas y columnas.
# dtypes       Indica el tipo de cada columna.
# info()       Resume la estructura y los valores no nulos.
# describe()   Calcula estadísticas descriptivas.
# mean()       Calcula la media aritmética.

# Las estadísticas permiten conocer el comportamiento de la comunidad, 
# pero no demuestran por sí mismas que una variable cause otra. 
# Por ejemplo, que los usuarios con más viajes compartidos 
# tengan mayor nivel de confianza en esta tabla no demuestra 
# una relación causal absoluta ni garantiza el mismo comportamiento en todas las rutas.