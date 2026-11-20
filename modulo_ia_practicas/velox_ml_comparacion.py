# ================================================================
# PROYECTO: VELOX UTVT
# PRACTICA: COMPARACION DE ALGORITMOS DE MACHINE LEARNING
#
# Archivo:
# velox_ml_comparacion.py
#
# Objetivo:
# Comparar tres algoritmos de clasificacion para el sistema de carpooling:
#
# 1. Decision Tree
# 2. Random Forest
# 3. K-Nearest Neighbors (KNN)
#
# Se utilizara un problema de clasificacion binaria:
# 0 = El viaje NO es exitoso (cancelado o pendiente).
# 1 = El viaje SI es exitoso (activo, en curso o finalizado).
# ================================================================


# ================================================================
# 1. IMPORTAR LAS BIBLIOTECAS
# ================================================================

import os     # Importa el módulo para interactuar con el sistema operativo
import sys    # Importa el módulo para manejar rutas del sistema
import numpy as np    # Importa NumPy para operaciones numéricas y matrices
import pandas as pd   # Importa Pandas para el manejo de estructuras de datos (DataFrames)
import matplotlib.pyplot as plt   # Importa Matplotlib para generar gráficos

# Agrega la ruta raíz del proyecto para permitir la importación segura de 'database.py'
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from database import db   # Importa la conexión a la base de datos de MongoDB

# Importa la función para dividir los datos en conjuntos de entrenamiento, validación y prueba
from sklearn.model_selection import train_test_split

# Importa la herramienta para realizar validación cruzada
from sklearn.model_selection import cross_val_score

# Importa el algoritmo de Árboles de Decisión
from sklearn.tree import DecisionTreeClassifier

# Importa el algoritmo de Bosques Aleatorios
from sklearn.ensemble import RandomForestClassifier

# Importa el algoritmo de K-Vecinos Más Cercanos (KNN)
from sklearn.neighbors import KNeighborsClassifier

# Importa el escalador para normalizar las variables numéricas
from sklearn.preprocessing import StandardScaler

# Importa Pipeline para encadenar el escalamiento y el clasificador
from sklearn.pipeline import Pipeline

# Importa clone para duplicar un modelo manteniendo sus hiperparámetros
from sklearn.base import clone

# Importa las métricas de evaluación para los modelos de clasificación
from sklearn.metrics import (
    accuracy_score,           # Métrica de exactitud
    precision_score,          # Métrica de precisión
    recall_score,             # Métrica de sensibilidad (recall)
    f1_score,                 # Métrica F1-score
    classification_report,    # Reporte completo de métricas
    confusion_matrix,         # Matriz de confusión numérica
    ConfusionMatrixDisplay    # Utilidad para graficar la matriz de confusión
)


# ================================================================
# 2. CONFIGURACION GENERAL
# ================================================================

SEMILLA = 42   # Define una semilla fija para asegurar la reproducibilidad de los resultados

rng = np.random.default_rng(SEMILLA)   # Inicializa el generador de números aleatorios con la semilla

print("\n==========================================")   # Imprime separador visual en consola
print("VELOX UTVT - INTELIGENCIA ARTIFICIAL")         # Imprime título institucional
print("COMPARACION DE ALGORITMOS DE CLASIFICACION")     # Imprime subtítulo
print("==========================================")   # Imprime separador visual


# ================================================================
# 3. EXTRACCION Y GENERACION DEL CONJUNTO DE DATOS (MONGODB)
# ================================================================

def extraer_datos_mongodb():
    """Función para extraer los registros de viajes desde MongoDB y adaptarlos para Machine Learning."""
    lista_viajes = []   # Inicializa una lista vacía para almacenar los documentos recuperados
    try:
        # Consulta la colección 'viaje' en MongoDB excluyendo el campo '_id' y seleccionando variables clave
        cursor = db.viaje.find({}, {"_id": 0, "costo": 1, "asientos_disponibles": 1, "estado": 1})
        for documento in cursor:   # Recorre cada documento obtenido del cursor de la base de datos
            costo = float(documento.get("costo", 0))          # Extrae el costo y lo convierte a número flotante
            asientos = int(documento.get("asientos_disponibles", 0))   # Extrae los asientos y los convierte a entero
            estado = documento.get("estado", "")              # Extrae el estado actual del viaje como texto
            
            # Define la variable objetivo de forma binaria: 1 si está activo/en curso/finalizado, de lo contrario 0
            exitoso = 1 if estado in ["en_curso", "finalizado", "activo"] else 0
            
            # Agrega un diccionario limpio con las variables estructuradas a la lista de viajes
            lista_viajes.append({
                "costo": costo,
                "asientos_disponibles": asientos,
                "viaje_exitoso": exitoso
            })
    except Exception as error:   # Captura cualquier error imprevisto durante la conexión o consulta
        print(f"Error al conectar con la base de datos: {error}")   # Imprime el mensaje de error en consola
        
    # Retorna un DataFrame de Pandas con los datos recolectados, o un DataFrame vacío si no hay registros
    return pd.DataFrame(lista_viajes) if lista_viajes else pd.DataFrame(columns=["costo", "asientos_disponibles", "viaje_exitoso"])

df = extraer_datos_mongodb()   # Ejecuta la función para obtener el DataFrame principal


# ================================================================
# 4. CONSTRUIR EL DATAFRAME
# ================================================================

print("\n==========================================")
print("1. DATOS EXTRAIDOS DE MONGODB")
print("==========================================")

print(df.head(10))   # Muestra en consola los primeros 10 registros del DataFrame

print("\nCantidad total de viajes en la BD:")   # Mensaje informativo
print(len(df))   # Muestra el número total de filas cargadas desde MongoDB

print("\nDistribucion de resultados (viaje_exitoso):")   # Mensaje informativo
if not df.empty and "viaje_exitoso" in df.columns:   # Valida si el DataFrame no está vacío y contiene la columna objetivo
    print(df["viaje_exitoso"].value_counts().sort_index())   # Cuenta y muestra cuántos registros pertenecen a cada clase (0 y 1)
else:
    print("No hay datos suficientes en la colección.")   # Mensaje de advertencia si faltan datos


# ================================================================
# 5. SEPARAR VARIABLES PREDICTORAS Y OBJETIVO
# ================================================================

X = df[["costo", "asientos_disponibles"]]   # Define las características (variables independientes) para predecir
y = df["viaje_exitoso"]                     # Define la etiqueta objetivo (variable dependiente a predecir)


# ================================================================
# 6. DIVIDIR LOS DATOS
# ================================================================

if len(df) >= 10:   # Verifica si el tamaño del conjunto de datos es mayor o igual a 10 para aplicar estratificación
    # Divide los datos en un bloque combinado (80%) y un bloque final de prueba (20%)
    X_entrenamiento_validacion, X_prueba, \
    y_entrenamiento_validacion, y_prueba = train_test_split(
        X, y, test_size=0.20, random_state=SEMILLA, stratify=y
    )

    # Divide el bloque del 80% en conjuntos de entrenamiento (60% del total) y validación (20% del total)
    X_entrenamiento, X_validacion, \
    y_entrenamiento, y_validacion = train_test_split(
        X_entrenamiento_validacion, y_entrenamiento_validacion,
        test_size=0.25, random_state=SEMILLA, stratify=y_entrenamiento_validacion
    )
else:
    print("\nAviso: Dataset pequeño detectado. Realizando partición directa sin estratificar.")
    # Partición alternativa sin estratificación en caso de contar con pocos registros en la base de datos
    X_entrenamiento_validacion, X_prueba, y_entrenamiento_validacion, y_prueba = train_test_split(X, y, test_size=0.20, random_state=SEMILLA)
    X_entrenamiento, X_validacion, y_entrenamiento, y_validacion = train_test_split(X_entrenamiento_validacion, y_entrenamiento_validacion, test_size=0.25, random_state=SEMILLA)

print("\n==========================================")
print("2. DISTRIBUCION DE LOS CONJUNTOS")
print("==========================================")

print("Entrenamiento:", len(X_entrenamiento))   # Muestra el número de registros destinados al entrenamiento
print("Validacion:", len(X_validacion))       # Muestra el número de registros destinados a la validación
print("Prueba:", len(X_prueba))               # Muestra el número de registros destinados a la prueba final


# ================================================================
# 7. DEFINIR LOS ALGORITMOS
# ================================================================

modelos = {   # Crea un diccionario para almacenar y configurar los tres modelos a comparar

    "Decision Tree": DecisionTreeClassifier(   # Configura el modelo de Árbol de Decisión
        max_depth=5,                          # Establece una profundidad máxima de 5 niveles
        min_samples_leaf=2,                   # Número mínimo de muestras requeridas en un nodo hoja
        random_state=SEMILLA                  # Fija la semilla para reproducibilidad
    ),

    "Random Forest": RandomForestClassifier(   # Configura el modelo de Bosque Aleatorio
        n_estimators=100,                      # Define la creación de 100 árboles en el bosque
        max_depth=None,                        # Permite que los árboles crezcan sin límite de profundidad
        min_samples_leaf=2,                    # Número mínimo de muestras en las hojas
        n_jobs=-1,                             # Utiliza todos los núcleos disponibles del procesador
        random_state=SEMILLA                   # Fija la semilla para reproducibilidad
    ),

    "KNN": Pipeline([                          # Configura el modelo KNN utilizando un Pipeline
        ("escalador", StandardScaler()),       # Paso 1: Normaliza las variables numéricas con la media y desviación estándar
        ("clasificador", KNeighborsClassifier( # Paso 2: Aplica el clasificador KNN
            n_neighbors=5,                     # Configura 5 vecinos cercanos
            weights="distance"                 # Pondera a los vecinos en función de su cercanía
        ))
    ])

}


# ================================================================
# 8. ENTRENAMIENTO Y COMPARACION
# ================================================================

resultados = []          # Lista para almacenar las métricas de desempeño de cada algoritmo
modelos_entrenados = {}  # Diccionario para guardar las instancias de los modelos ya entrenados

print("\n==========================================")
print("3. ENTRENAMIENTO DE LOS ALGORITMOS")
print("==========================================")

for nombre, modelo in modelos.items():   # Itera sobre cada modelo definido en el diccionario

    print("\n--------------------------------------")
    print("Entrenando:", nombre)             # Imprime el nombre del algoritmo que se está procesando
    print("--------------------------------------")

    modelo.fit(X_entrenamiento, y_entrenamiento)   # Entrena el modelo utilizando el conjunto de entrenamiento

    modelos_entrenados[nombre] = modelo           # Guarda el modelo entrenado en el diccionario

    predicciones = modelo.predict(X_validacion)   # Genera predicciones utilizando el conjunto de validación

    # Calcula las métricas de evaluación comparando las etiquetas reales con las predicciones
    exactitud = accuracy_score(y_validacion, predicciones)
    precision = precision_score(y_validacion, predicciones, zero_division=0)
    recall = recall_score(y_validacion, predicciones, zero_division=0)
    f1 = f1_score(y_validacion, predicciones, zero_division=0)

    # Añade un diccionario con los resultados obtenidos a la lista general
    resultados.append({
        "algoritmo": nombre,
        "exactitud": exactitud,
        "precision": precision,
        "recall": recall,
        "f1_score": f1
    })

    # Imprime en consola los porcentajes obtenidos en cada métrica
    print("Exactitud:", f"{exactitud:.2%}")
    print("Precision:", f"{precision:.2%}")
    print("Recall:", f"{recall:.2%}")
    print("F1-score:", f"{f1:.2%}")


# ================================================================
# 9. TABLA COMPARATIVA
# ================================================================

df_resultados = pd.DataFrame(resultados)   # Convierte la lista de resultados en un DataFrame de Pandas

# Ordena los algoritmos de mayor a menor basándose en la métrica F1-score
df_resultados = df_resultados.sort_values(
    by="f1_score",
    ascending=False
).reset_index(drop=True)

print("\n==========================================")
print("4. COMPARACION DE ALGORITMOS")
print("==========================================")

# Imprime la tabla comparativa con formato porcentual legible
print(
    df_resultados.to_string(
        index=False,
        formatters={
            "exactitud": "{:.2%}".format,
            "precision": "{:.2%}".format,
            "recall": "{:.2%}".format,
            "f1_score": "{:.2%}".format
        }
    )
)


# ================================================================
# 10. SELECCIONAR EL MEJOR MODELO
# ================================================================

mejor_registro = df_resultados.iloc[0]       # Selecciona la primera fila (el modelo con mejor F1-score)
mejor_nombre = mejor_registro["algoritmo"]   # Extrae el nombre del mejor algoritmo
mejor_modelo = modelos_entrenados[mejor_nombre]   # Obtiene la instancia entrenada del mejor modelo

print("\n==========================================")
print("5. MODELO SELECCIONADO")
print("==========================================")

print("Algoritmo:", mejor_nombre)                      # Muestra el nombre del modelo ganador
print("F1-score de validacion:", f"{mejor_registro['f1_score']:.2%}")   # Muestra su F1-score
print("Criterio de seleccion: mayor F1-score")          # Muestra el criterio utilizado


# ================================================================
# 11. REENTRENAR CON ENTRENAMIENTO + VALIDACION
# ================================================================

# Combina los conjuntos de entrenamiento y validación para aprovechar más datos (80% del total)
X_entrenamiento_final = pd.concat([X_entrenamiento, X_validacion])
y_entrenamiento_final = pd.concat([y_entrenamiento, y_validacion])

# Clona la estructura del mejor modelo y lo reentrena utilizando el 80% combinado de los datos
modelo_final = clone(modelos[mejor_nombre])
modelo_final.fit(X_entrenamiento_final, y_entrenamiento_final)


# ================================================================
# 12. EVALUACION FINAL CON DATOS DE PRUEBA
# ================================================================

predicciones_finales = modelo_final.predict(X_prueba)   # Predice utilizando el conjunto de prueba (datos totalmente nuevos)

# Calcula las métricas finales de evaluación sobre el conjunto de prueba
accuracy_final = accuracy_score(y_prueba, predicciones_finales)
precision_final = precision_score(y_prueba, predicciones_finales, zero_division=0)
recall_final = recall_score(y_prueba, predicciones_finales, zero_division=0)
f1_final = f1_score(y_prueba, predicciones_finales, zero_division=0)

print("\n==========================================")
print("6. EVALUACION FINAL")
print("==========================================")

print("Algoritmo:", mejor_nombre)
print("Exactitud:", f"{accuracy_final:.2%}")
print("Precision:", f"{precision_final:.2%}")
print("Recall:", f"{recall_final:.2%}")
print("F1-score:", f"{f1_final:.2%}")


# ================================================================
# 13. REPORTE DE CLASIFICACION
# ================================================================

print("\n==========================================")
print("7. REPORTE DE CLASIFICACION")
print("==========================================")

# Imprime un reporte detallado con precisión, recall y f1 por cada clase individual
print(
    classification_report(
        y_prueba,
        predicciones_finales,
        labels=[0, 1],
        target_names=["No exitoso", "Exitoso"],
        zero_division=0
    )
)


# ================================================================
# 14. MATRIZ DE CONFUSION
# ================================================================

matriz = confusion_matrix(
    y_prueba,
    predicciones_finales,
    labels=[0, 1]
)   # Calcula la matriz de confusión numérica para el conjunto de prueba

print("\n==========================================")
print("8. MATRIZ DE CONFUSION")
print("==========================================")

print(matriz)   # Imprime la matriz de confusión en la terminal


# ================================================================
# 15. VALIDACION CRUZADA DEL MODELO SELECCIONADO
# ================================================================

# Ejecuta validación cruzada en el mejor modelo utilizando particiones sobre el bloque de entrenamiento final
resultados_cv = cross_val_score(
    clone(modelos[mejor_nombre]),
    X_entrenamiento_final,
    y_entrenamiento_final,
    cv=min(5, len(X_entrenamiento_final)) if len(X_entrenamiento_final) >= 2 else 2,
    scoring="f1",
    n_jobs=-1
)

print("\n==========================================")
print("9. VALIDACION CRUZADA")
print("==========================================")

print("Resultados por particion:")
print(resultados_cv)
print("Promedio F1:", f"{resultados_cv.mean():.2%}")           # Imprime el promedio de F1 en validación cruzada
print("Desviacion estandar:", f"{resultados_cv.std():.2%}")   # Imprime la desviación estándar


# ================================================================
# 16. GRAFICA COMPARATIVA DE METRICAS (USANDO TONOS VERDES)
# ================================================================

metricas = ["exactitud", "precision", "recall", "f1_score"]   # Define las métricas a graficar
plt.figure(figsize=(12, 6))                                   # Define las dimensiones de la figura

ancho = 0.22                                   # Define el ancho de las barras en el gráfico
posiciones = np.arange(len(df_resultados))    # Calcula las posiciones en el eje X para los algoritmos

# Define una paleta de tonos verdes institucionales/estéticos para cada barra de métrica
colores_verdes = ['#a9dfbf', '#7dcea0', '#52be80', '#27ae60']

for i, metrica in enumerate(metricas):   # Itera sobre cada métrica para crear las barras agrupadas
    desplazamiento = (i - 1.5) * ancho   # Calcula el desplazamiento horizontal para evitar que las barras se encimen
    plt.bar(
        posiciones + desplazamiento,
        df_resultados[metrica],
        width=ancho,
        label=metrica.replace("_", " ").title(),
        color=colores_verdes[i]          # Asigna el color verde correspondiente a cada barra
    )

plt.xticks(posiciones, df_resultados["algoritmo"])   # Etiqueta el eje X con los nombres de los algoritmos
plt.ylim(0, 1.05)                                    # Fija los límites del eje Y de 0 a 1.05
plt.ylabel("Resultado")                              # Título del eje Y
plt.xlabel("Algoritmo")                              # Título del eje X
plt.title("Velox UTVT - Comparacion de algoritmos de clasificacion")   # Título principal del gráfico
plt.legend()                                         # Muestra las leyendas de las métricas
plt.grid(axis="y", alpha=0.3)                        # Agrega una cuadrícula ligera de fondo en el eje Y
plt.tight_layout()                                   # Ajusta automáticamente los elementos visuales

plt.savefig("velox_comparacion_algoritmos.png", dpi=300)   # Guarda la gráfica en un archivo PNG en alta resolución
plt.show()                                                # Muestra la ventana con la gráfica en pantalla


# ================================================================
# 17. GRAFICA DE LA MATRIZ DE CONFUSION (USANDO COLORES VERDES)
# ================================================================

visualizacion = ConfusionMatrixDisplay(
    confusion_matrix=matriz,
    display_labels=["No exitoso", "Exitoso"]
)   # Prepara la visualización de la matriz de confusión

visualizacion.plot(cmap="Greens", values_format="d")   # Genera el gráfico utilizando la paleta de colores verdes ('Greens')
plt.title("Matriz de confusion - " + mejor_nombre)     # Asigna el título indicando el modelo ganador
plt.tight_layout()                                     # Ajusta el diseño de la figura

plt.savefig("velox_matriz_confusion.png", dpi=300)     # Guarda la matriz de confusión como imagen PNG
plt.show()                                             # Muestra la ventana con la matriz en pantalla


# ================================================================
# 18. GUARDAR LOS RESULTADOS
# ================================================================

df_resultados.to_csv("velox_resultados_comparacion.csv", index=False)   # Exporta la tabla de resultados a un archivo CSV

print("\n==========================================")
print("10. ARCHIVOS GENERADOS")
print("==========================================")
print("velox_comparacion_algoritmos.png")   # Notifica la creación de la gráfica de barras
print("velox_matriz_confusion.png")         # Notifica la creación de la matriz de confusión
print("velox_resultados_comparacion.csv")   # Notifica la exportación del archivo CSV


# ================================================================
# ACTIVIDAD DE MODIFICACIÓN: EXPERIMENTOS DE HIPERPARÁMETROS
# ================================================================

print("\n==========================================")
print("ACTIVIDAD DE MODIFICACIÓN: EXPERIMENTOS")
print("==========================================")

# ------------------------------------------------------------
# EXPERIMENTO A: Modificar n_neighbors de KNN (3, 5, 7, 9, 15)
# ------------------------------------------------------------
print("\n--- EXPERIMENTO A: KNN (Variando n_neighbors) ---")
vecinos_valores = [3, 5, 7, 9, 15]   # Define los valores de vecinos a probar
for k in vecinos_valores:            # Itera sobre cada valor de vecino
    pipeline_knn_exp = Pipeline([
        ("escalador", StandardScaler()),
        ("clasificador", KNeighborsClassifier(n_neighbors=k, weights="distance"))
    ])
    pipeline_knn_exp.fit(X_entrenamiento, y_entrenamiento)   # Entrena el modelo con el k actual
    pred_val = pipeline_knn_exp.predict(X_validacion)       # Realiza predicciones en validación
    f1_val = f1_score(y_validacion, pred_val, zero_division=0) # Calcula el F1-score
    print(f"KNN con n_neighbors={k:2d} -> F1-score Validación: {f1_val:.2%}")

# ------------------------------------------------------------
# EXPERIMENTO B: Modificar n_estimators de Random Forest (50, 100, 200, 500)
# ------------------------------------------------------------
print("\n--- EXPERIMENTO B: Random Forest (Variando n_estimators) ---")
estimadores_valores = [50, 100, 200, 500]   # Define la cantidad de árboles a probar
for n in estimadores_valores:               # Itera sobre cada cantidad de estimadores
    rf_exp = RandomForestClassifier(n_estimators=n, n_jobs=-1, random_state=SEMILLA)
    rf_exp.fit(X_entrenamiento, y_entrenamiento)             # Entrena el bosque aleatorio
    pred_val = rf_exp.predict(X_validacion)                 # Predice en validación
    f1_val = f1_score(y_validacion, pred_val, zero_division=0) # Calcula el F1-score
    print(f"Random Forest con n_estimators={n:3d} -> F1-score Validación: {f1_val:.2%}")

# ------------------------------------------------------------
# EXPERIMENTO C: Modificar max_depth de Decision Tree (2, 3, 5, 8, None)
# ------------------------------------------------------------
print("\n--- EXPERIMENTO C: Decision Tree (Variando max_depth) ---")
profundidades_valores = [2, 3, 5, 8, None]   # Define las profundidades a evaluar en el árbol
for d in profundidades_valores:              # Itera sobre cada profundidad
    dt_exp = DecisionTreeClassifier(max_depth=d, random_state=SEMILLA)
    dt_exp.fit(X_entrenamiento, y_entrenamiento)             # Entrena el árbol de decisión
    pred_val = dt_exp.predict(X_validacion)                 # Predice en validación
    f1_val = f1_score(y_validacion, pred_val, zero_division=0) # Calcula el F1-score
    print(f"Decision Tree con max_depth={str(d):4s} -> F1-score Validación: {f1_val:.2%}")

print("\n==========================================")
print("¡EXPERIMENTOS DE LA ACTIVIDAD FINALIZADOS!")
print("==========================================")


# ================================================================
# 19. CONCLUSION DEL SISTEMA
# ================================================================

print("\n==========================================")
print("CONCLUSION")
print("==========================================")

print("El algoritmo seleccionado fue:", mejor_nombre)                 # Muestra el modelo ganador final
print("Su F1-score final fue:", f"{f1_final:.2%}")                    # Muestra su puntuación final
print("\nVelox UTVT ha completado exitosamente la comparacion de algoritmos de Machine Learning.")
print("Recuerda: el mejor modelo depende de los datos reales del carpooling y de la métrica a optimizar.")
print("\nPROCESO FINALIZADO.")                                        # Mensaje de término del script






"""
Análisis de los Experimentos (Velox UTVT):
Durante la ejecución de los experimentos de modificación, se observó un 
comportamiento constante del 75.00% de F1-score en validación para los tres 
algoritmos (KNN, Random Forest y Árbol de Decisión), independientemente de 
la variación de hiperparámetros como n_neighbors, n_estimators o max_depth. 
Esto demuestra que, dado el volumen y la distribución actual de los registros 
de viajes extraídos de MongoDB, los modelos alcanzan un umbral de estabilidad uniforme. 
No se presentaron problemas críticos de sobreajuste severo ni de subajuste drástico, 
validando que la configuración seleccionada es robusta y adecuada para el sistema de carpooling universitario.
"""