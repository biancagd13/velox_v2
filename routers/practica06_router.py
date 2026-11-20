import os
import matplotlib
matplotlib.use('Agg')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.templating import Jinja2Templates
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV, StratifiedKFold
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)
from scipy.stats import randint

from database import db

router = APIRouter(prefix="/ia/practica06", tags=["IA - Práctica 06"])
templates = Jinja2Templates(directory="templates")

os.makedirs("static/img/ia", exist_ok=True)
os.makedirs("resultados", exist_ok=True)
SEMILLA = 42


def extraer_datos_mongodb():
    lista_viajes = []
    try:
        cursor = db.viaje.find({}, {"_id": 0, "costo": 1, "asientos_disponibles": 1, "estado": 1})
        for documento in cursor:
            costo = float(documento.get("costo", 0))
            asientos = int(documento.get("asientos_disponibles", 0))
            estado = documento.get("estado", "")
            exitoso = 1 if estado in ["en_curso", "finalizado", "activo"] else 0
            lista_viajes.append({
                "costo": costo,
                "asientos_disponibles": asientos,
                "viaje_exitoso": exitoso
            })
    except Exception as error:
        print(f"Error al conectar con MongoDB: {error}")
    return pd.DataFrame(lista_viajes) if lista_viajes else pd.DataFrame(columns=["costo", "asientos_disponibles", "viaje_exitoso"])


@router.get("/", response_class=HTMLResponse)
async def vista_practica06(request: Request):
    df = extraer_datos_mongodb()

    if len(df) < 5:
        return templates.TemplateResponse(
            request,
            "ia_modulo/practica06.html",
            {"request": request, "error": "No hay suficientes registros en MongoDB (mínimo 5 viajes) para ejecutar la optimización."}
        )

    X = df[["costo", "asientos_disponibles"]]
    y = df["viaje_exitoso"]

    if len(df) >= 10 and y.value_counts().min() >= 2:
        X_entrenamiento, X_prueba, y_entrenamiento, y_prueba = train_test_split(
            X, y, test_size=0.20, random_state=SEMILLA, stratify=y
        )
    else:
        X_entrenamiento, X_prueba, y_entrenamiento, y_prueba = train_test_split(
            X, y, test_size=0.20, random_state=SEMILLA
        )

    n_splits_dinamico = min(5, len(X_entrenamiento)) if len(X_entrenamiento) >= 2 else 2
    validacion_cruzada = StratifiedKFold(n_splits=n_splits_dinamico, shuffle=True, random_state=SEMILLA)

    modelo_arbol = DecisionTreeClassifier(random_state=SEMILLA)
    parametros_arbol = {
        "max_depth": [2, 3, 5, 8, 10, None],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4],
        "criterion": ["gini", "entropy"]
    }

    modelo_bosque = RandomForestClassifier(random_state=SEMILLA, n_jobs=-1)
    parametros_bosque = {
        "n_estimators": [50, 100, 200],
        "max_depth": [3, 5, 8, None],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4]
    }

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

    resultados_grid = []
    mejores_modelos_grid = {}

    for nombre, (modelo, parametros) in modelos.items():
        busqueda = GridSearchCV(
            estimator=modelo,
            param_grid=parametros,
            scoring="f1",
            cv=validacion_cruzada,
            n_jobs=-1
        )
        busqueda.fit(X_entrenamiento, y_entrenamiento)
        mejores_modelos_grid[nombre] = busqueda.best_estimator_
        resultados_grid.append({
            "algoritmo": nombre,
            "mejor_f1": round(busqueda.best_score_ * 100, 2),
            "hiperparametros": str(busqueda.best_params_),
            "combinaciones": len(busqueda.cv_results_["params"])
        })

    df_grid = pd.DataFrame(resultados_grid)

    resultados_random = []
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

    for nombre, (modelo, _) in modelos.items():
        busqueda_random = RandomizedSearchCV(
            estimator=modelo,
            param_distributions=parametros_random[nombre],
            n_iter=min(10, len(X_entrenamiento) * 2),
            scoring="f1",
            cv=validacion_cruzada,
            n_jobs=-1,
            random_state=SEMILLA
        )
        busqueda_random.fit(X_entrenamiento, y_entrenamiento)
        resultados_random.append({
            "algoritmo": nombre,
            "mejor_f1": round(busqueda_random.best_score_ * 100, 2),
            "hiperparametros": str(busqueda_random.best_params_)
        })

    df_random = pd.DataFrame(resultados_random)

    comparacion = pd.merge(
        df_grid[["algoritmo", "mejor_f1"]].rename(columns={"mejor_f1": "gridsearch"}),
        df_random[["algoritmo", "mejor_f1"]].rename(columns={"mejor_f1": "randomizedsearch"}),
        on="algoritmo"
    )

    mejor_indice = df_grid["mejor_f1"].idxmax()
    mejor_nombre = df_grid.loc[mejor_indice, "algoritmo"]
    mejor_modelo = mejores_modelos_grid[mejor_nombre]

    predicciones = mejor_modelo.predict(X_prueba)
    accuracy = accuracy_score(y_prueba, predicciones)
    precision = precision_score(y_prueba, predicciones, zero_division=0)
    recall = recall_score(y_prueba, predicciones, zero_division=0)
    f1 = f1_score(y_prueba, predicciones, zero_division=0)

    metricas_finales = {
        "algoritmo": mejor_nombre,
        "accuracy": round(accuracy * 100, 2),
        "precision": round(precision * 100, 2),
        "recall": round(recall * 100, 2),
        "f1": round(f1 * 100, 2)
    }

    reporte_str = classification_report(
        y_prueba, predicciones, labels=[0, 1], target_names=["No exitoso", "Exitoso"], zero_division=0
    )
    matriz = confusion_matrix(y_prueba, predicciones, labels=[0, 1]).tolist()

    # Gráfica comparativa de optimización
    posiciones = np.arange(len(comparacion))
    ancho = 0.35
    plt.figure(figsize=(9, 4.5))
    plt.bar(posiciones - ancho / 2, comparacion["gridsearch"], width=ancho, label="GridSearchCV", color="#27ae60")
    plt.bar(posiciones + ancho / 2, comparacion["randomizedsearch"], width=ancho, label="RandomizedSearchCV", color="#a9dfbf")
    plt.xticks(posiciones, comparacion["algoritmo"])
    plt.ylim(0, 105)
    plt.ylabel("F1-score (%)")
    plt.xlabel("Algoritmo")
    plt.title("Velox UTVT - Comparacion de metodos de optimizacion")
    plt.legend()
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    ruta_grafica_opt = "static/img/ia/p06_comparacion_optimizacion.png"
    plt.savefig(ruta_grafica_opt, dpi=150)
    plt.close()

    # Matriz de confusión
    matriz_arr = confusion_matrix(y_prueba, predicciones, labels=[0, 1])
    plt.figure(figsize=(5, 4.5))
    plt.imshow(matriz_arr, cmap="Greens")
    plt.title(f"Matriz de confusion - {mejor_nombre}")
    plt.xlabel("Prediccion del modelo")
    plt.ylabel("Resultado real")
    plt.xticks([0, 1], ["No exitoso", "Exitoso"])
    plt.yticks([0, 1], ["No exitoso", "Exitoso"])
    for f in range(2):
        for c in range(2):
            color_texto = "white" if matriz_arr[f, c] > matriz_arr.max() / 2 else "black"
            plt.text(c, f, str(matriz_arr[f, c]), ha="center", va="center", fontsize=14, color=color_texto, fontweight="bold")
    plt.colorbar()
    plt.tight_layout()
    ruta_grafica_matriz = "static/img/ia/p06_matriz.png"
    plt.savefig(ruta_grafica_matriz, dpi=150)
    plt.close()

    df_grid.to_csv("resultados/p06_resultados_gridsearch.csv", index=False)
    df_random.to_csv("resultados/p06_resultados_randomizedsearch.csv", index=False)
    comparacion.to_csv("resultados/p06_comparacion_metodos.csv", index=False)

    return templates.TemplateResponse(
        request,
        "ia_modulo/practica06.html",
        {
            "request": request,
            "total_registros": len(df),
            "conteos": {"entrenamiento": len(X_entrenamiento), "prueba": len(X_prueba)},
            "resultados_grid": df_grid.to_dict(orient="records"),
            "comparacion": comparacion.to_dict(orient="records"),
            "metricas": metricas_finales,
            "reporte_str": reporte_str,
            "matriz": matriz,
            "img_opt": "/" + ruta_grafica_opt,
            "img_matriz": "/" + ruta_grafica_matriz
        }
    )


@router.get("/exportar_csv/{tipo}")
async def exportar_csv(tipo: str):
    rutas = {
        "grid": "resultados/p06_resultados_gridsearch.csv",
        "random": "resultados/p06_resultados_randomizedsearch.csv",
        "comparacion": "resultados/p06_comparacion_metodos.csv"
    }
    ruta = rutas.get(tipo, rutas["grid"])
    if os.path.exists(ruta):
        return FileResponse(ruta, media_type="text/csv", filename=os.path.basename(ruta))
    return {"error": "Archivo no encontrado."}
