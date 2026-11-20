# ==============================================================================
# PROYECTO: VELOX UTVT
# PRACTICA 04: ENTRENAMIENTO, VALIDACION Y PRUEBA
#
# Objetivo:
# Comprender como evaluar un modelo de Machine Learning mediante la division
# metodologica (60% entrenamiento, 20% validacion y 20% prueba) y detectar
# problemas de sobreajuste (overfitting) y subajuste (underfitting).
#
# Variables extraidas de MongoDB para el sistema de carpooling:
# - costo: Costo monetario del viaje
# - asientos_disponibles: Espacios libres en el vehiculo
# - viaje_exitoso: 1 = Exitoso (activo/en_curso/finalizado), 0 = No exitoso
# ==============================================================================


# ------------------------------------------------------------
# 1. IMPORTAR BIBLIOTECAS
# ------------------------------------------------------------

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Permitir importar 'database.py' desde la raiz del proyecto
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from database import db

# Herramienta para separar los conjuntos de datos.
from sklearn.model_selection import train_test_split

# Herramienta para validacion cruzada.
from sklearn.model_selection import cross_val_score

# Modelo de clasificacion (Arbol de Decision).
from sklearn.tree import DecisionTreeClassifier

# Metricas de evaluacion.
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)


# ------------------------------------------------------------
# 2. CREAR CARPETA DE RESULTADOS
# ------------------------------------------------------------

os.makedirs("resultados", exist_ok=True)


# ------------------------------------------------------------
# 3. EXTRACCION DE DATOS DESDE MONGODB (VELOX UTVT)
# ------------------------------------------------------------

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

print("\n==========================================")
print("1. DATOS EXTRAIDOS DE MONGODB")
print("==========================================")

print(df.head(10))

print("\nCantidad total de registros:")
print(len(df))

print("\nDistribucion de las clases (viaje_exitoso):")
if not df.empty and "viaje_exitoso" in df.columns:
    print(df["viaje_exitoso"].value_counts())
else:
    print("No hay datos suficientes.")


# ------------------------------------------------------------
# 4. DEFINIR VARIABLES DE ENTRADA Y OBJETIVO
# ------------------------------------------------------------

X = df[["costo", "asientos_disponibles"]]
y = df["viaje_exitoso"]

SEMILLA = 42


# ------------------------------------------------------------
# 5. SEPARAR ENTRENAMIENTO Y DATOS TEMPORALES (60% / 40%)
# ------------------------------------------------------------

if len(df) >= 10:
    X_entrenamiento, X_temporal, y_entrenamiento, y_temporal = train_test_split(
        X, y, test_size=0.40, random_state=SEMILLA, stratify=y
    )
    
    # ------------------------------------------------------------
    # 6. DIVIDIR VALIDACION Y PRUEBA (50% del temporal c/u -> 20% / 20%)
    # ------------------------------------------------------------
    X_validacion, X_prueba, y_validacion, y_prueba = train_test_split(
        X_temporal, y_temporal, test_size=0.50, random_state=SEMILLA, stratify=y_temporal
    )
else:
    print("\nAviso: Dataset pequeño detectado. Realizando partición directa sin estratificar.")
    X_entrenamiento, X_temporal, y_entrenamiento, y_temporal = train_test_split(X, y, test_size=0.40, random_state=SEMILLA)
    X_validacion, X_prueba, y_validacion, y_prueba = train_test_split(X_temporal, y_temporal, test_size=0.50, random_state=SEMILLA)

print("\n==========================================")
print("2. DISTRIBUCION DE LOS CONJUNTOS")
print("==========================================")

print("Entrenamiento:", len(X_entrenamiento))
print("Validacion:", len(X_validacion))
print("Prueba:", len(X_prueba))


# ------------------------------------------------------------
# 7. ENTRENAR MODELOS CON DIFERENTES PROFUNDIDADES
# ------------------------------------------------------------

profundidades = [1, 2, 3, 5, 10, None]
resultados = []
modelos = {}

for profundidad in profundidades:

    print("\n--------------------------------------")
    print("Entrenando modelo con profundidad:", profundidad)
    print("--------------------------------------")

    modelo = DecisionTreeClassifier(
        max_depth=profundidad,
        random_state=SEMILLA
    )

    modelo.fit(X_entrenamiento, y_entrenamiento)

    pred_entrenamiento = modelo.predict(X_entrenamiento)
    pred_validacion = modelo.predict(X_validacion)

    exactitud_entrenamiento = accuracy_score(y_entrenamiento, pred_entrenamiento)
    exactitud_validacion = accuracy_score(y_validacion, pred_validacion)

    modelos[profundidad] = modelo

    resultados.append({
        "profundidad": (
            "Sin limite"
            if profundidad is None
            else str(profundidad)
        ),
        "profundidad_grafica": (
            12
            if profundidad is None
            else profundidad
        ),
        "entrenamiento": exactitud_entrenamiento,
        "validacion": exactitud_validacion
    })

df_resultados = pd.DataFrame(resultados)

print("\n==========================================")
print("3. COMPARACION DE MODELOS")
print("==========================================")

print(
    df_resultados[
        ["profundidad", "entrenamiento", "validacion"]
    ].to_string(index=False, formatters={"entrenamiento": "{:.2%}".format, "validacion": "{:.2%}".format})
)


# ------------------------------------------------------------
# 8. IDENTIFICAR POSIBLES PROBLEMAS DE AJUSTE
# ------------------------------------------------------------

print("\n==========================================")
print("4. ANALISIS DE AJUSTE")
print("==========================================")

for _, fila in df_resultados.iterrows():
    diferencia = fila["entrenamiento"] - fila["validacion"]
    profundidad = fila["profundidad"]

    print(f"\nModelo profundidad: {profundidad}")
    print(f"Exactitud entrenamiento: {fila['entrenamiento']:.2%}")
    print(f"Exactitud validacion: {fila['validacion']:.2%}")
    print(f"Diferencia: {diferencia:.2%}")

    if fila["entrenamiento"] < 0.60 and fila["validacion"] < 0.60:
        print("-> Posible subajuste (Underfitting).")
    elif diferencia > 0.15:
        print("-> Posible sobreajuste (Overfitting).")
    else:
        print("-> Buen ajuste o sin brecha critica.")


# ------------------------------------------------------------
# 9. SELECCIONAR EL MODELO CON MEJOR VALIDACION
# ------------------------------------------------------------

indice_mejor = df_resultados["validacion"].idxmax()
mejor_registro = df_resultados.loc[indice_mejor]

profundidad_seleccionada = int(mejor_registro["profundidad_grafica"])
if profundidad_seleccionada == 12:
    profundidad_seleccionada = None

mejor_modelo = modelos[profundidad_seleccionada]

print("\n==========================================")
print("5. MODELO SELECCIONADO")
print("==========================================")

print("Profundidad seleccionada:", mejor_registro["profundidad"])
print(f"Exactitud de validacion: {mejor_registro['validacion']:.2%}")


# ------------------------------------------------------------
# 10. EVALUACION FINAL CON EL CONJUNTO DE PRUEBA
# ------------------------------------------------------------

pred_prueba = mejor_modelo.predict(X_prueba)

exactitud = accuracy_score(y_prueba, pred_prueba)
precision = precision_score(y_prueba, pred_prueba, zero_division=0)
recall = recall_score(y_prueba, pred_prueba, zero_division=0)
f1 = f1_score(y_prueba, pred_prueba, zero_division=0)

print("\n==========================================")
print("6. EVALUACION FINAL")
print("==========================================")

print(f"Exactitud: {exactitud:.2%}")
print(f"Precision: {precision:.2%}")
print(f"Recall: {recall:.2%}")
print(f"F1-score: {f1:.2%}")


# ------------------------------------------------------------
# 11. REPORTE DE CLASIFICACION
# ------------------------------------------------------------

print("\n==========================================")
print("7. REPORTE DE CLASIFICACION")
print("==========================================")

print(
    classification_report(
        y_prueba,
        pred_prueba,
        labels=[0, 1],
        target_names=["No exitoso", "Exitoso"],
        zero_division=0
    )
)


# ------------------------------------------------------------
# 12. MATRIZ DE CONFUSION
# ------------------------------------------------------------

matriz = confusion_matrix(
    y_prueba,
    pred_prueba,
    labels=[0, 1]
)

print("\n==========================================")
print("8. MATRIZ DE CONFUSION")
print("==========================================")

print(matriz)


# ------------------------------------------------------------
# 13. VALIDACION CRUZADA
# ------------------------------------------------------------

modelo_cv = DecisionTreeClassifier(
    max_depth=profundidad_seleccionada,
    random_state=SEMILLA
)

puntuaciones_cv = cross_val_score(
    modelo_cv,
    X_entrenamiento,
    y_entrenamiento,
    cv=min(5, len(X_entrenamiento)) if len(X_entrenamiento) >= 2 else 2,
    scoring="accuracy"
)

print("\n==========================================")
print("9. VALIDACION CRUZADA")
print("==========================================")

print("Resultados de cada particion:")
print(puntuaciones_cv)
print(f"Promedio: {puntuaciones_cv.mean():.2%}")
print(f"Desviacion estandar: {puntuaciones_cv.std():.2%}")


# ------------------------------------------------------------
# 14. GRAFICA: ENTRENAMIENTO VS VALIDACION (Con desfase visual)
# ------------------------------------------------------------

plt.figure(figsize=(11, 6))
x = df_resultados["profundidad_grafica"]

# Pequeño desfase estético para que no se oculten las líneas si tienen el mismo valor
offset = 0.012
y_entrenamiento_visual = df_resultados["entrenamiento"] + offset
y_validacion_visual = df_resultados["validacion"] - offset

plt.plot(x, y_entrenamiento_visual, marker="o", linewidth=2.5, linestyle="-", color="tab:blue", label="Entrenamiento")
plt.plot(x, y_validacion_visual, marker="s", linewidth=2.5, linestyle="--", color="tab:orange", label="Validacion")

plt.xticks(x, df_resultados["profundidad"])
plt.title("Velox UTVT - Comparacion del rendimiento (Entrenamiento vs Validacion)", fontsize=13)
plt.xlabel("Profundidad del arbol")
plt.ylabel("Exactitud (con ajuste visual)")
plt.ylim(0, 1.1)
plt.grid(True, linestyle="--", alpha=0.5)
plt.legend()
plt.tight_layout()

plt.savefig("resultados/velox_comparacion_modelos.png", dpi=300, bbox_inches="tight")
plt.show()


# ------------------------------------------------------------
# 15. GRAFICA: MATRIZ DE CONFUSION
# ------------------------------------------------------------

plt.figure(figsize=(7, 6))
plt.imshow(matriz, cmap="Blues")
plt.title("Matriz de confusion - Velox UTVT")
plt.xlabel("Prediccion del modelo")
plt.ylabel("Resultado real")
plt.xticks([0, 1], ["No exitoso", "Exitoso"])
plt.yticks([0, 1], ["No exitoso", "Exitoso"])

for fila in range(2):
    for columna in range(2):
        plt.text(columna, fila, str(matriz[fila, columna]), ha="center", va="center", fontsize=14)

plt.colorbar()
plt.tight_layout()
plt.savefig("resultados/velox_matriz_confusion.png", dpi=300)
plt.show()


# ------------------------------------------------------------
# 16. EXPORTAR RESULTADOS
# ------------------------------------------------------------

df_resultados.to_csv("resultados/velox_comparacion_modelos.csv", index=False)

df_predicciones = X_prueba.copy()
df_predicciones["resultado_real"] = y_prueba
df_predicciones["prediccion"] = pred_prueba
df_predicciones.to_csv("resultados/velox_predicciones_prueba.csv", index=False)


# ------------------------------------------------------------
# 17. CONCLUSIONES AUTOMATICAS
# ------------------------------------------------------------

print("\n==========================================")
print("10. CONCLUSIONES")
print("==========================================")

print("El entrenamiento permite aprender patrones a partir de los datos reales de MongoDB.")
print("La validacion ayuda a seleccionar la profundidad optima sin tocar los datos de prueba.")
print("La prueba evalua el comportamiento final con datos totalmente nuevos.")
print("Una brecha alta entre entrenamiento y validacion indica sobreajuste (overfitting).")
print("\nArchivos generados en la carpeta 'resultados/'.")
print("\nPRACTICA 04 ADAPTADA FINALIZADA.")