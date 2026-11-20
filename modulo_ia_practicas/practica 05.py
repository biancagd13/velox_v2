# ================================================================
# PROYECTO: VELOX UTVT
# PRACTICA 05: COMPARACION DE ALGORITMOS DE MACHINE LEARNING
#
# Archivo:
# velox_ml_comparacion_practica05.py
#
# Objetivo:
# Comparar tres algoritmos de clasificacion para el sistema de carpooling:
# 1. Decision Tree (Árbol de Decisión)
# 2. Random Forest (Bosque Aleatorio)
# 3. K-Nearest Neighbors (KNN)
#
# Se utilizara un problema de clasificacion binaria basado en MongoDB:
# 0 = El viaje NO es exitoso (cancelado o pendiente).
# 1 = El viaje SI es exitoso (activo, en curso o finalizado).
# ================================================================


# ================================================================
# 1. IMPORTAR LAS BIBLIOTECAS
# ================================================================

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Herramienta para asegurar la importación de 'database.py' desde la raíz del proyecto
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from database import db

# train_test_split divide los datos en entrenamiento, validacion y prueba.
from sklearn.model_selection import train_test_split

# cross_val_score realiza validacion cruzada.
from sklearn.model_selection import cross_val_score

# DecisionTreeClassifier implementa arboles de decision.
from sklearn.tree import DecisionTreeClassifier

# RandomForestClassifier implementa bosques aleatorios.
from sklearn.ensemble import RandomForestClassifier

# KNeighborsClassifier implementa el algoritmo KNN.
from sklearn.neighbors import KNeighborsClassifier

# StandardScaler normaliza las variables numericas.
from sklearn.preprocessing import StandardScaler

# Pipeline permite unir el escalamiento y el algoritmo.
from sklearn.pipeline import Pipeline

# clone permite crear una copia de un modelo conservando su configuracion.
from sklearn.base import clone

# Metricas para evaluar los modelos.
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)


# ================================================================
# 2. CONFIGURACION GENERAL
# ================================================================

# Fijamos una semilla para obtener resultados reproducibles.
SEMILLA = 42

# Generador de numeros aleatorios.
rng = np.random.default_rng(SEMILLA)

print("\n==========================================")
print("VELOX UTVT - INTELIGENCIA ARTIFICIAL")
print("COMPARACION DE ALGORITMOS DE CLASIFICACION")
print("==========================================")


# ================================================================
# 3. EXTRACCION Y GENERACION DEL CONJUNTO DE DATOS (MONGODB)
# ================================================================

def extraer_datos_mongodb():
    """Función para extraer los registros de viajes desde MongoDB y adaptarlos para Machine Learning."""
    lista_viajes = []
    try:
        # Consultamos la colección 'viaje' en MongoDB Atlas
        cursor = db.viaje.find({}, {"_id": 0, "costo": 1, "asientos_disponibles": 1, "estado": 1})
        for documento in cursor:
            costo = float(documento.get("costo", 0))
            asientos = int(documento.get("asientos_disponibles", 0))
            estado = documento.get("estado", "")
            
            # Definición de la variable objetivo binaria
            exitoso = 1 if estado in ["en_curso", "finalizado", "activo"] else 0
            
            lista_viajes.append({
                "costo": costo,
                "asientos_disponibles": asientos,
                "viaje_exitoso": exitoso
            })
    except Exception as error:
        print(f"Error al conectar con la base de datos: {error}")
        
    return pd.DataFrame(lista_viajes) if lista_viajes else pd.DataFrame(columns=["costo", "asientos_disponibles", "viaje_exitoso"])

df = extraer_datos_mongodb()


# ================================================================
# 4. CONSTRUIR EL DATAFRAME
# ================================================================

print("\n==========================================")
print("1. DATOS EXTRAIDOS DE MONGODB")
print("==========================================")

# Mostramos los primeros diez registros.
print(df.head(10))

# Mostramos la cantidad de registros.
print("\nCantidad total de viajes en la BD:")
print(len(df))

# Mostramos cuantos viajes son exitosos y cuantos no.
print("\nDistribucion de resultados (viaje_exitoso):")
if not df.empty and "viaje_exitoso" in df.columns:
    print(df["viaje_exitoso"].value_counts().sort_index())
else:
    print("No hay datos suficientes en la colección.")


# ================================================================
# 5. SEPARAR VARIABLES PREDICTORAS Y OBJETIVO
# ================================================================

# X contiene las variables que utilizaran los algoritmos para sus predicciones.
X = df[["costo", "asientos_disponibles"]]

# y contiene el resultado que queremos predecir.
y = df["viaje_exitoso"]


# ================================================================
# 6. DIVIDIR LOS DATOS
# ================================================================

# Verificamos si tenemos suficientes datos para aplicar estratificación segura
if len(df) >= 10:
    # Primera division: 80% entrenamiento + validacion / 20% prueba.
    X_entrenamiento_validacion, X_prueba, \
    y_entrenamiento_validacion, y_prueba = train_test_split(
        X, y, test_size=0.20, random_state=SEMILLA, stratify=y
    )

    # Segunda division del 80%: 75% entrenamiento / 25% validacion.
    # Resultado total: 60% entrenamiento, 20% validacion, 20% prueba.
    X_entrenamiento, X_validacion, \
    y_entrenamiento, y_validacion = train_test_split(
        X_entrenamiento_validacion, y_entrenamiento_validacion,
        test_size=0.25, random_state=SEMILLA, stratify=y_entrenamiento_validacion
    )
else:
    print("\nAviso: Dataset pequeño detectado. Realizando partición directa sin estratificar.")
    X_entrenamiento_validacion, X_prueba, y_entrenamiento_validacion, y_prueba = train_test_split(X, y, test_size=0.20, random_state=SEMILLA)
    X_entrenamiento, X_validacion, y_entrenamiento, y_validacion = train_test_split(X_entrenamiento_validacion, y_entrenamiento_validacion, test_size=0.25, random_state=SEMILLA)

print("\n==========================================")
print("2. DISTRIBUCION DE LOS CONJUNTOS")
print("==========================================")

print("Entrenamiento:", len(X_entrenamiento))
print("Validacion:", len(X_validacion))
print("Prueba:", len(X_prueba))


# ================================================================
# 7. DEFINIR LOS ALGORITMOS
# ================================================================

# Creamos un diccionario para almacenar los modelos.
modelos = {

    # ------------------------------------------------------------
    # MODELO 1: ARBOL DE DECISION
    # ------------------------------------------------------------
    "Decision Tree": DecisionTreeClassifier(
        max_depth=5,
        min_samples_leaf=2,
        random_state=SEMILLA
    ),

    # ------------------------------------------------------------
    # MODELO 2: RANDOM FOREST
    # ------------------------------------------------------------
    "Random Forest": RandomForestClassifier(
        n_estimators=100,
        max_depth=None,
        min_samples_leaf=2,
        n_jobs=-1,
        random_state=SEMILLA
    ),

    # ------------------------------------------------------------
    # MODELO 3: K-NEAREST NEIGHBORS
    # ------------------------------------------------------------
    # KNN utiliza Pipeline para normalizar las variables numéricas primero.
    "KNN": Pipeline([
        ("escalador", StandardScaler()),
        ("clasificador", KNeighborsClassifier(
            n_neighbors=5,
            weights="distance"
        ))
    ])

}


# ================================================================
# 8. ENTRENAMIENTO Y COMPARACION
# ================================================================

resultados = []
modelos_entrenados = {}

print("\n==========================================")
print("3. ENTRENAMIENTO DE LOS ALGORITMOS")
print("==========================================")

for nombre, modelo in modelos.items():

    print("\n--------------------------------------")
    print("Entrenando:", nombre)
    print("--------------------------------------")

    # Entrenamos el algoritmo utilizando exclusivamente el conjunto de entrenamiento.
    modelo.fit(X_entrenamiento, y_entrenamiento)

    # Guardamos el modelo entrenado.
    modelos_entrenados[nombre] = modelo

    # Realizamos predicciones con los datos de validacion.
    predicciones = modelo.predict(X_validacion)

    # Calculamos las metricas de evaluacion.
    exactitud = accuracy_score(y_validacion, predicciones)
    precision = precision_score(y_validacion, predicciones, zero_division=0)
    recall = recall_score(y_validacion, predicciones, zero_division=0)
    f1 = f1_score(y_validacion, predicciones, zero_division=0)

    # Guardamos los resultados.
    resultados.append({
        "algoritmo": nombre,
        "exactitud": exactitud,
        "precision": precision,
        "recall": recall,
        "f1_score": f1
    })

    print("Exactitud:", f"{exactitud:.2%}")
    print("Precision:", f"{precision:.2%}")
    print("Recall:", f"{recall:.2%}")
    print("F1-score:", f"{f1:.2%}")


# ================================================================
# 9. TABLA COMPARATIVA
# ================================================================

df_resultados = pd.DataFrame(resultados)

# Ordenamos los algoritmos por su F1-score para seleccionar el mejor modelo.
df_resultados = df_resultados.sort_values(
    by="f1_score",
    ascending=False
).reset_index(drop=True)

print("\n==========================================")
print("4. COMPARACION DE ALGORITMOS")
print("==========================================")

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

mejor_registro = df_resultados.iloc[0]
mejor_nombre = mejor_registro["algoritmo"]
mejor_modelo = modelos_entrenados[mejor_nombre]

print("\n==========================================")
print("5. MODELO SELECCIONADO")
print("==========================================")

print("Algoritmo:", mejor_nombre)
print("F1-score de validacion:", f"{mejor_registro['f1_score']:.2%}")
print("Criterio de seleccion: mayor F1-score")


# ================================================================
# 11. REENTRENAR CON ENTRENAMIENTO + VALIDACION
# ================================================================

X_entrenamiento_final = pd.concat([X_entrenamiento, X_validacion])
y_entrenamiento_final = pd.concat([y_entrenamiento, y_validacion])

# Creamos una copia independiente del modelo seleccionado y lo reentrenamos con el 80% de los datos.
modelo_final = clone(modelos[mejor_nombre])
modelo_final.fit(X_entrenamiento_final, y_entrenamiento_final)


# ================================================================
# 12. EVALUACION FINAL CON DATOS DE PRUEBA
# ================================================================

predicciones_finales = modelo_final.predict(X_prueba)

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
)

print("\n==========================================")
print("8. MATRIZ DE CONFUSION")
print("==========================================")

print(matriz)


# ================================================================
# 15. VALIDACION CRUZADA DEL MODELO SELECCIONADO
# ================================================================

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
print("Promedio F1:", f"{resultados_cv.mean():.2%}")
print("Desviacion estandar:", f"{resultados_cv.std():.2%}")


# ================================================================
# 16. GRAFICA COMPARATIVA DE METRICAS (TONOS VERDES)
# ================================================================

metricas = ["exactitud", "precision", "recall", "f1_score"]
plt.figure(figsize=(12, 6))

ancho = 0.22
posiciones = np.arange(len(df_resultados))
colores_verdes = ['#a9dfbf', '#7dcea0', '#52be80', '#27ae60']

for i, metrica in enumerate(metricas):
    desplazamiento = (i - 1.5) * ancho
    plt.bar(
        posiciones + desplazamiento,
        df_resultados[metrica],
        width=ancho,
        label=metrica.replace("_", " ").title(),
        color=colores_verdes[i]
    )

plt.xticks(posiciones, df_resultados["algoritmo"])
plt.ylim(0, 1.05)
plt.ylabel("Resultado")
plt.xlabel("Algoritmo")
plt.title("Velox UTVT - Comparacion de algoritmos de clasificacion")
plt.legend()
plt.grid(axis="y", alpha=0.3)
plt.tight_layout()

plt.savefig("velox_comparacion_algoritmos.png", dpi=300)
plt.show()


# ================================================================
# 17. GRAFICA DE LA MATRIZ DE CONFUSION (COLORES VERDES)
# ================================================================

visualizacion = ConfusionMatrixDisplay(
    confusion_matrix=matriz,
    display_labels=["No exitoso", "Exitoso"]
)

visualizacion.plot(cmap="Greens", values_format="d")
plt.title("Matriz de confusion - " + mejor_nombre)
plt.tight_layout()

plt.savefig("velox_matriz_confusion.png", dpi=300)
plt.show()


# ================================================================
# 18. GUARDAR LOS RESULTADOS
# ================================================================

df_resultados.to_csv("velox_resultados_comparacion.csv", index=False)

print("\n==========================================")
print("10. ARCHIVOS GENERADOS")
print("==========================================")
print("velox_comparacion_algoritmos.png")
print("velox_matriz_confusion.png")
print("velox_resultados_comparacion.csv")


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
vecinos_valores = [3, 5, 7, 9, 15]
for k in vecinos_valores:
    pipeline_knn_exp = Pipeline([
        ("escalador", StandardScaler()),
        ("clasificador", KNeighborsClassifier(n_neighbors=k, weights="distance"))
    ])
    pipeline_knn_exp.fit(X_entrenamiento, y_entrenamiento)
    pred_val = pipeline_knn_exp.predict(X_validacion)
    f1_val = f1_score(y_validacion, pred_val, zero_division=0)
    print(f"KNN con n_neighbors={k:2d} -> F1-score Validación: {f1_val:.2%}")

# ------------------------------------------------------------
# EXPERIMENTO B: Modificar n_estimators de Random Forest (50, 100, 200, 500)
# ------------------------------------------------------------
print("\n--- EXPERIMENTO B: Random Forest (Variando n_estimators) ---")
estimadores_valores = [50, 100, 200, 500]
for n in estimadores_valores:
    rf_exp = RandomForestClassifier(n_estimators=n, n_jobs=-1, random_state=SEMILLA)
    rf_exp.fit(X_entrenamiento, y_entrenamiento)
    pred_val = rf_exp.predict(X_validacion)
    f1_val = f1_score(y_validacion, pred_val, zero_division=0)
    print(f"Random Forest con n_estimators={n:3d} -> F1-score Validación: {f1_val:.2%}")

# ------------------------------------------------------------
# EXPERIMENTO C: Modificar max_depth de Decision Tree (2, 3, 5, 8, None)
# ------------------------------------------------------------
print("\n--- EXPERIMENTO C: Decision Tree (Variando max_depth) ---")
profundidades_valores = [2, 3, 5, 8, None]
for d in profundidades_valores:
    dt_exp = DecisionTreeClassifier(max_depth=d, random_state=SEMILLA)
    dt_exp.fit(X_entrenamiento, y_entrenamiento)
    pred_val = dt_exp.predict(X_validacion)
    f1_val = f1_score(y_validacion, pred_val, zero_division=0)
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

print("El algoritmo seleccionado fue:", mejor_nombre)
print("Su F1-score final fue:", f"{f1_final:.2%}")
print("\nVelox UTVT ha completado exitosamente la comparacion de algoritmos de Machine Learning.")
print("Recuerda: el mejor modelo depende de los datos reales del carpooling y de la métrica a optimizar.")
print("\nPROCESO FINALIZADO.")


"""
Análisis de los Experimentos (Velox UTVT):
Durante la ejecución de los experimentos de modificación, se observó un 
comportamiento constante en el rendimiento de validación para los tres 
algoritmos (KNN, Random Forest y Árbol de Decisión), independientemente de 
la variación de hiperparámetros como n_neighbors, n_estimators o max_depth. 
Esto demuestra que, dado el volumen y la distribución actual de los registros 
de viajes extraídos de MongoDB, los modelos alcanzan un umbral de estabilidad uniforme. 
No se presentaron problemas críticos de sobreajuste severo ni de subajuste drástico, 
validando que la configuración seleccionada es robusta y adecuada para el sistema de carpooling universitario.
"""