# ============================================================
# PRACTICA 2 ADAPTADA A VELOX UTVT - PARTE D
# LIMPIEZA BASICA Y EXPORTACION
# ============================================================

# Estrategia de limpieza
# Para este ejercicio utilizaremos una estrategia sencilla:
# - Eliminar filas completamente duplicadas.
# - Conservar los valores faltantes como tales durante la inspección.
# - Marcar los registros con valores fuera de rango.
# - Excluir los registros incompletos o inválidos de un archivo de práctica, 
# explicando por qué se excluyeron.

# En un proyecto real, eliminar filas no siempre es la mejor opción. 
# La estrategia debe elegirse según la cantidad de datos, 
# la causa de los valores faltantes y el objetivo del análisis.

import os
import pandas as pd

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

# ------------------------------------------------------------
# PASO 1. ELIMINAR FILAS COMPLETAMENTE DUPLICADAS
# ------------------------------------------------------------
df_limpio = df.drop_duplicates().copy()

# ------------------------------------------------------------
# PASO 2. IDENTIFICAR REGISTROS VALIDOS
# ------------------------------------------------------------
# Comprobar que no existan valores faltantes en las columnas
# requeridas y que los porcentajes y calificaciones estén
# dentro de sus rangos permitidos.

columnas_requeridas = [
    "usuario",
    "viajes_compartidos",
    "puntualidad",
    "calificacion_conductor"
]

completo = df_limpio[columnas_requeridas].notna().all(axis=1)

puntualidad_valida = df_limpio["puntualidad"].between(0, 100)

calificacion_valida = df_limpio["calificacion_conductor"].between(0, 100)

viajes_validos = df_limpio["viajes_compartidos"] >= 0

# Combinar las condiciones.
valido = (
    completo
    & puntualidad_valida
    & calificacion_valida
    & viajes_validos
)

# ------------------------------------------------------------
# PASO 3. SEPARAR LOS REGISTROS VALIDOS
# ------------------------------------------------------------
df_validos = df_limpio.loc[valido].copy()

# Los registros descartados se conservan para poder auditarlos.
df_revision = df_limpio.loc[~valido].copy()

print("=== REGISTROS VALIDOS DE VELOX ===")
print(df_validos)

print("\n=== REGISTROS DE VELOX QUE REQUIEREN REVISION ===")
print(df_revision)

# ------------------------------------------------------------
# PASO 4. EXPORTAR LOS RESULTADOS (DE FORMA SEGURA)
# ------------------------------------------------------------
carpeta_actual = os.path.dirname(os.path.abspath(__file__))
ruta_limpios = os.path.join(carpeta_actual, "usuarios_limpios.csv")
ruta_revision = os.path.join(carpeta_actual, "usuarios_revision.csv")

df_validos.to_csv(
    ruta_limpios,
    index=False,
    encoding="utf-8"
)

df_revision.to_csv(
    ruta_revision,
    index=False,
    encoding="utf-8"
)

print("\nArchivos generados en el módulo:")
print(f"- {ruta_limpios}")
print(f"- {ruta_revision}")


# Ahora tenemos dos archivos:
# - usuarios_limpios.csv: contiene los registros que cumplen las condiciones definidas para el ejercicio en Velox.
# - usuarios_revision.csv: conserva los registros que requieren revisión por errores o nulos.
# Esta separación ayuda a evitar que los datos problemáticos de la plataforma desaparezcan sin explicación. 

# En un sistema profesional, también convendría registrar el motivo de exclusión y 
# revisar los valores con la persona responsable de los datos.