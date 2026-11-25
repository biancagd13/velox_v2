# ================================================================
# PROYECTO: VELOX UTVT
# PRACTICA 06: OPTIMIZACION DE HIPERPARAMETROS
#
# Archivo:
# velox_ml_optimizacion_practica06.py
#
# Objetivo:
# Encontrar las mejores configuraciones de los algoritmos
# Decision Tree, Random Forest y KNN mediante GridSearchCV y RandomizedSearchCV
# utilizando datos reales extraídos de MongoDB para el sistema de carpooling.
#
# Variables extraidas de MongoDB:
# - costo: Costo monetario del viaje
# - asientos_disponibles: Espacios libres en el vehiculo
# - viaje_exitoso: 1 = Exitoso (activo/en_curso/finalizado), 0 = No exitoso
# ================================================================


# ================================================================
# 1. IMPORTAR BIBLIOTECAS
# ================================================================

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Permitir importar 'database.py' desde la raiz del proyecto
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from database import db

# Separacion de datos.
from sklearn.model_selection import train_test_split

# Busqueda exhaustiva y aleatoria de hiperparametros.
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

# Validacion cruzada estratificada.
from sklearn.model_selection import StratifiedKFold

# Algoritmos de clasificacion.
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

# Normalizacion y Pipeline.
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# Metricas de evaluacion.
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)

# Distribuciones para la busqueda aleatoria.
from scipy.stats import randint


# ================================================================
# 2. CONFIGURACION GENERAL Y DIRECTORIOS
# ================================================================

SEMILLA = 42
rng = np.random.default_rng(SEMILLA)

os.makedirs("resultados", exist_ok=True)

print("\n==========================================")
print("VELOX UTVT - INTELIGENCIA ARTIFICIAL")
print("OPTIMIZACION DE HIPERPARAMETROS")
print("==========================================")


# ================================================================
# 3. EXTRACCION DE DATOS DESDE MONGODB (VELOX UTVT)
# ================================================================

def extraer_datos_mongodb():
    lista_viajes = []
    try:
        cursor = db.viaje.find({}, {"_id": 0, "costo": 1, "asientos_disponibles": 1, "estado": 1})
        for documento in cursor:
            costo = float(documento.get("costo", 0))
            asientos = int(documento.get("asientos_disponibles", 0))
            estado = documento.get("estado", "")
            
            # Definicion de la variable objetivo binaria
            exitoso = 1 if estado in ["en_curso", "finalizado", "activo"] else 0
            
            lista_viajes.append({
                "costo": costo,
                "asientos_disponibles": asientos,
                "viaje_exitoso": exitoso
            })
    except Exception as error:
        print(f"Error al conectar con MongoDB: {error}")
        
    return pd.DataFrame(lista_viajes) if lista_viajes else pd.DataFrame(columns=["costo", "asientos_disponibles", "viaje_exitoso"])

df = extraer_datos_mongodb()


# ================================================================
# 4. CREAR Y MOSTRAR DATAFRAME
# ================================================================

print("\n==========================================")
print("1. DATOS EXTRAIDOS DE MONGODB")
print("==========================================")

print(df.head(10))
print("\nCantidad total de viajes en la BD:", len(df))

print("\nDistribucion de clases (viaje_exitoso):")
if not df.empty and "viaje_exitoso" in df.columns:
    print(df["viaje_exitoso"].value_counts().sort_index())
else:
    print("No hay datos suficientes.")


# ================================================================
# 5. SEPARAR VARIABLES PREDICTORAS Y OBJETIVO
# ================================================================

X = df[["costo", "asientos_disponibles"]]
y = df["viaje_exitoso"]


# ================================================================
# 6. DIVIDIR DATOS EN ENTRENAMIENTO Y PRUEBA (80% / 20%)
# ================================================================

if len(df) >= 10 and y.value_counts().min() >= 2:
    X_entrenamiento, X_prueba, y_entrenamiento, y_prueba = train_test_split(
        X, y, test_size=0.20, random_state=SEMILLA, stratify=y
    )
else:
    print("\nAviso: Dataset pequeño o desbalanceado. Realizando partición sin estratificar.")
    X_entrenamiento, X_prueba, y_entrenamiento, y_prueba = train_test_split(
        X, y, test_size=0.20, random_state=SEMILLA
    )

print("\n==========================================")
print("2. DISTRIBUCION DE DATOS")
print("==========================================")
print("Entrenamiento:", len(X_entrenamiento))
print("Prueba:", len(X_prueba))


# ================================================================
# 7. CONFIGURAR VALIDACION CRUZADA ESTRATIFICADA
# ================================================================

# Ajustamos n_splits dinámicamente si el conjunto de entrenamiento es pequeño
n_splits_dinamico = min(5, len(X_entrenamiento)) if len(X_entrenamiento) >= 2 else 2

validacion_cruzada = StratifiedKFold(
    n_splits=n_splits_dinamico,
    shuffle=True,
    random_state=SEMILLA
)


# ================================================================
# 8. DEFINIR LOS ALGORITMOS Y SUS HIPERPARAMETROS
# ================================================================

# Modelo 1: Decision Tree
modelo_arbol = DecisionTreeClassifier(random_state=SEMILLA)
parametros_arbol = {
    "max_depth": [2, 3, 5, 8, 10, None],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 4],
    "criterion": ["gini", "entropy"]
}

# Modelo 2: Random Forest
modelo_bosque = RandomForestClassifier(random_state=SEMILLA, n_jobs=-1)
parametros_bosque = {
    "n_estimators": [50, 100, 200],
    "max_depth": [3, 5, 8, None],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 4]
}

# Modelo 3: KNN (con Pipeline para normalizar)
modelo_knn = Pipeline([
    ("escalador", StandardScaler()),
    ("clasificador", KNeighborsClassifier())
])
parametros_knn = {
    "clasificador__n_neighbors": [3, 5, 7, 9, 11, 15],
    "clasificador__weights": ["uniform", "distance"],
    "clasificador__metric": ["euclidean", "manhattan"]
}

modelos = {
    "Decision Tree": (modelo_arbol, parametros_arbol),
    "Random Forest": (modelo_bosque, parametros_bosque),
    "KNN": (modelo_knn, parametros_knn)
}


# ================================================================
# 9. EJECUTAR GRIDSEARCHCV
# ================================================================

resultados_grid = []
mejores_modelos_grid = {}

print("\n==========================================")
print("3. OPTIMIZACION CON GRIDSEARCHCV")
print("==========================================")

for nombre, configuracion in modelos.items():
    modelo, parametros = configuracion
    print(f"\n--------------------------------------")
    print(f"Optimizando con GridSearch: {nombre}")
    print(f"--------------------------------------")

    busqueda = GridSearchCV(
        estimator=modelo,
        param_grid=parametros,
        scoring="f1",
        cv=validacion_cruzada,
        n_jobs=-1,
        return_train_score=True,
        verbose=0
    )

    busqueda.fit(X_entrenamiento, y_entrenamiento)
    mejores_modelos_grid[nombre] = busqueda.best_estimator_

    print("Mejor F1-score:", f"{busqueda.best_score_:.2%}")
    print("Mejores hiperparámetros:", busqueda.best_params_)

    resultados_grid.append({
        "algoritmo": nombre,
        "mejor_f1": busqueda.best_score_,
        "hiperparametros": str(busqueda.best_params_),
        "combinaciones": len(busqueda.cv_results_["params"])
    })

df_grid = pd.DataFrame(resultados_grid)


# ================================================================
# 10. RANDOMIZEDSEARCHCV
# ================================================================

resultados_random = []
mejores_modelos_random = {}

print("\n==========================================")
print("4. OPTIMIZACION CON RANDOMIZEDSEARCHCV")
print("==========================================")

parametros_random = {
    "Decision Tree": {
        "max_depth": randint(2, 15),
        "min_samples_split": randint(2, 15),
        "min_samples_leaf": randint(1, 8),
        "criterion": ["gini", "entropy"]
    },
    "Random Forest": {
        "n_estimators": randint(50, 301),
        "max_depth": [3, 5, 8, 12, None],
        "min_samples_split": randint(2, 12),
        "min_samples_leaf": randint(1, 6)
    },
    "KNN": {
        "clasificador__n_neighbors": [3, 5, 7, 9, 11, 15, 19],
        "clasificador__weights": ["uniform", "distance"],
        "clasificador__metric": ["euclidean", "manhattan"]
    }
}

for nombre, configuracion in modelos.items():
    modelo, _ = configuracion
    print(f"\n--------------------------------------")
    print(f"Optimizando aleatoriamente: {nombre}")
    print(f"--------------------------------------")

    busqueda_random = RandomizedSearchCV(
        estimator=modelo,
        param_distributions=parametros_random[nombre],
        n_iter=min(10, len(X_entrenamiento) * 2), # Ajuste iterativo seguro para datasets pequeños
        scoring="f1",
        cv=validacion_cruzada,
        n_jobs=-1,
        random_state=SEMILLA,
        verbose=0
    )

    busqueda_random.fit(X_entrenamiento, y_entrenamiento)
    mejores_modelos_random[nombre] = busqueda_random.best_estimator_

    print("Mejor F1-score:", f"{busqueda_random.best_score_:.2%}")
    print("Mejores hiperparámetros:", busqueda_random.best_params_)

    resultados_random.append({
        "algoritmo": nombre,
        "mejor_f1": busqueda_random.best_score_,
        "hiperparametros": str(busqueda_random.best_params_)
    })

df_random = pd.DataFrame(resultados_random)


# ================================================================
# 11. COMPARAR GRIDSEARCH Y RANDOMIZEDSEARCH
# ================================================================

print("\n==========================================")
print("5. COMPARACION DE METODOS DE OPTIMIZACION")
print("==========================================")

comparacion = pd.merge(
    df_grid[["algoritmo", "mejor_f1"]].rename(columns={"mejor_f1": "gridsearch"}),
    df_random[["algoritmo", "mejor_f1"]].rename(columns={"mejor_f1": "randomizedsearch"}),
    on="algoritmo"
)

print(
    comparacion.to_string(
        index=False,
        formatters={
            "gridsearch": "{:.2%}".format,
            "randomizedsearch": "{:.2%}".format
        }
    )
)


# ================================================================
# 12. SELECCIONAR EL MEJOR ALGORITMO
# ================================================================

mejor_indice = df_grid["mejor_f1"].idxmax()
mejor_nombre = df_grid.loc[mejor_indice, "algoritmo"]
mejor_modelo = mejores_modelos_grid[mejor_nombre]

print("\n==========================================")
print("6. MEJOR MODELO OPTIMIZADO")
print("==========================================")
print("Algoritmo:", mejor_nombre)
print(f"F1-score en validacion cruzada: {df_grid.loc[mejor_indice, 'mejor_f1']:.2%}")
print("\nConfiguracion seleccionada:")
print(mejor_modelo.get_params())


# ================================================================
# 13. EVALUACION FINAL CON CONJUNTO DE PRUEBA
# ================================================================

predicciones = mejor_modelo.predict(X_prueba)

accuracy = accuracy_score(y_prueba, predicciones)
precision = precision_score(y_prueba, predicciones, zero_division=0)
recall = recall_score(y_prueba, predicciones, zero_division=0)
f1 = f1_score(y_prueba, predicciones, zero_division=0)

print("\n==========================================")
print("7. EVALUACION CON DATOS DE PRUEBA")
print("==========================================")
print("Accuracy:", f"{accuracy:.2%}")
print("Precision:", f"{precision:.2%}")
print("Recall:", f"{recall:.2%}")
print("F1-score:", f"{f1:.2%}")


# ================================================================
# 14. REPORTE DE CLASIFICACION
# ================================================================

print("\n==========================================")
print("8. REPORTE DE CLASIFICACION")
print("==========================================")
print(
    classification_report(
        y_prueba,
        predicciones,
        labels=[0, 1],
        target_names=["No exitoso", "Exitoso"],
        zero_division=0
    )
)


# ================================================================
# 15. MATRIZ DE CONFUSION
# ================================================================

matriz = confusion_matrix(y_prueba, predicciones, labels=[0, 1])

print("\n==========================================")
print("9. MATRIZ DE CONFUSION")
print("==========================================")
print(matriz)

visualizacion = ConfusionMatrixDisplay(
    confusion_matrix=matriz,
    display_labels=["No exitoso", "Exitoso"]
)
visualizacion.plot(cmap="Greens", values_format="d")
plt.title("Matriz de confusion - " + mejor_nombre)
plt.tight_layout()
plt.savefig("resultados/p06_matriz_confusion.png", dpi=300)
plt.show()


# ================================================================
# 16. GRAFICA COMPARATIVA DE OPTIMIZACION (TONOS VERDES)
# ================================================================

posiciones = np.arange(len(comparacion))
ancho = 0.35

plt.figure(figsize=(10, 6))
plt.bar(
    posiciones - ancho / 2,
    comparacion["gridsearch"],
    width=ancho,
    label="GridSearchCV",
    color="#27ae60"
)
plt.bar(
    posiciones + ancho / 2,
    comparacion["randomizedsearch"],
    width=ancho,
    label="RandomizedSearchCV",
    color="#a9dfbf"
)

plt.xticks(posiciones, comparacion["algoritmo"])
plt.ylim(0, 1.05)
plt.ylabel("F1-score")
plt.xlabel("Algoritmo")
plt.title("Velox UTVT - Comparacion de metodos de optimizacion")
plt.legend()
plt.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig("resultados/p06_comparacion_optimizacion.png", dpi=300)
plt.show()


# ================================================================
# 17. EXPORTAR RESULTADOS A CSV
# ================================================================

df_grid.to_csv("resultados/p06_resultados_gridsearch.csv", index=False, encoding="utf-8")
df_random.to_csv("resultados/p06_resultados_randomizedsearch.csv", index=False, encoding="utf-8")
comparacion.to_csv("resultados/p06_comparacion_metodos.csv", index=False, encoding="utf-8")


# ================================================================
# 18. CONCLUSION
# ================================================================

print("\n==========================================")
print("CONCLUSION DEL PROYECTO")
print("==========================================")
print("Mejor algoritmo:", mejor_nombre)
print("Accuracy final:", f"{accuracy:.2%}")
print("Precision final:", f"{precision:.2%}")
print("Recall final:", f"{recall:.2%}")
print("F1-score final:", f"{f1:.2%}")
print("\nArchivos generados en la carpeta 'resultados/':")
print("- p06_matriz_confusion.png")
print("- p06_comparacion_optimizacion.png")
print("- p06_resultados_gridsearch.csv")
print("- p06_resultados_randomizedsearch.csv")
print("- p06_comparacion_metodos.csv")
print("\nPRACTICA 06 FINALIZADA.")