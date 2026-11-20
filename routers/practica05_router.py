import os
import matplotlib
matplotlib.use('Agg')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.templating import Jinja2Templates
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.base import clone
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)

from database import db

router = APIRouter(prefix="/ia/practica05", tags=["IA - Práctica 05"])
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
        print(f"Error al conectar con la base de datos: {error}")
    return pd.DataFrame(lista_viajes) if lista_viajes else pd.DataFrame(columns=["costo", "asientos_disponibles", "viaje_exitoso"])


@router.get("/", response_class=HTMLResponse)
async def vista_practica05(request: Request):
    df = extraer_datos_mongodb()

    if len(df) < 5:
        return templates.TemplateResponse(
            request,
            "ia_modulo/practica05.html",
            {"request": request, "error": "No hay suficientes registros en MongoDB (mínimo 5 viajes)."}
        )

    X = df[["costo", "asientos_disponibles"]]
    y = df["viaje_exitoso"]

    # División de datos
    if len(df) >= 10 and y.value_counts().min() >= 3:
        X_ent_val, X_prueba, y_ent_val, y_prueba = train_test_split(X, y, test_size=0.20, random_state=SEMILLA, stratify=y)
        X_entrenamiento, X_validacion, y_entrenamiento, y_validacion = train_test_split(X_ent_val, y_ent_val, test_size=0.25, random_state=SEMILLA, stratify=y_ent_val)
    else:
        X_ent_val, X_prueba, y_ent_val, y_prueba = train_test_split(X, y, test_size=0.20, random_state=SEMILLA)
        X_entrenamiento, X_validacion, y_entrenamiento, y_validacion = train_test_split(X_ent_val, y_ent_val, test_size=0.25, random_state=SEMILLA)

    modelos = {
        "Decision Tree": DecisionTreeClassifier(max_depth=5, min_samples_leaf=2, random_state=SEMILLA),
        "Random Forest": RandomForestClassifier(n_estimators=100, max_depth=None, min_samples_leaf=2, n_jobs=-1, random_state=SEMILLA),
        "KNN": Pipeline([
            ("escalador", StandardScaler()),
            ("clasificador", KNeighborsClassifier(n_neighbors=5, weights="distance"))
        ])
    }

    resultados = []
    modelos_entrenados = {}

    for nombre, modelo in modelos.items():
        modelo.fit(X_entrenamiento, y_entrenamiento)
        modelos_entrenados[nombre] = modelo
        predicciones = modelo.predict(X_validacion)

        exactitud = accuracy_score(y_validacion, predicciones)
        precision = precision_score(y_validacion, predicciones, zero_division=0)
        recall = recall_score(y_validacion, predicciones, zero_division=0)
        f1 = f1_score(y_validacion, predicciones, zero_division=0)

        resultados.append({
            "algoritmo": nombre,
            "exactitud": round(exactitud * 100, 2),
            "precision": round(precision * 100, 2),
            "recall": round(recall * 100, 2),
            "f1_score": round(f1 * 100, 2)
        })

    df_resultados = pd.DataFrame(resultados).sort_values(by="f1_score", ascending=False).reset_index(drop=True)

    # Mejor modelo
    mejor_registro = df_resultados.iloc[0]
    mejor_nombre = mejor_registro["algoritmo"]

    # Reentrenar con 80%
    X_entrenamiento_final = pd.concat([X_entrenamiento, X_validacion])
    y_entrenamiento_final = pd.concat([y_entrenamiento, y_validacion])

    modelo_final = clone(modelos[mejor_nombre])
    modelo_final.fit(X_entrenamiento_final, y_entrenamiento_final)

    # Evaluación final
    predicciones_finales = modelo_final.predict(X_prueba)
    accuracy_final = accuracy_score(y_prueba, predicciones_finales)
    precision_final = precision_score(y_prueba, predicciones_finales, zero_division=0)
    recall_final = recall_score(y_prueba, predicciones_finales, zero_division=0)
    f1_final = f1_score(y_prueba, predicciones_finales, zero_division=0)

    metricas_finales = {
        "algoritmo": mejor_nombre,
        "exactitud": round(accuracy_final * 100, 2),
        "precision": round(precision_final * 100, 2),
        "recall": round(recall_final * 100, 2),
        "f1": round(f1_final * 100, 2)
    }

    # Reporte de clasificación en texto
    reporte_str = classification_report(
        y_prueba, predicciones_finales, labels=[0, 1], target_names=["No exitoso", "Exitoso"], zero_division=0
    )

    # Matriz de confusión numérica
    matriz = confusion_matrix(y_prueba, predicciones_finales, labels=[0, 1]).tolist()

    # Validación cruzada detallada
    cv_folds = min(5, len(X_entrenamiento_final)) if len(X_entrenamiento_final) >= 2 else 2
    resultados_cv = cross_val_score(
        clone(modelos[mejor_nombre]), X_entrenamiento_final, y_entrenamiento_final, cv=cv_folds, scoring="f1", n_jobs=-1
    )
    cv_info = {
        "particiones": [round(val * 100, 2) for val in resultados_cv],
        "promedio": round(resultados_cv.mean() * 100, 2),
        "desviacion": round(resultados_cv.std() * 100, 2)
    }

    # --- EXPERIMENTOS ---
    experimentos = {"knn": [], "rf": [], "dt": []}

    for k in [3, 5, 7, 9, 15]:
        pipe = Pipeline([("esc", StandardScaler()), ("clf", KNeighborsClassifier(n_neighbors=k, weights="distance"))])
        pipe.fit(X_entrenamiento, y_entrenamiento)
        f1_val = f1_score(y_validacion, pipe.predict(X_validacion), zero_division=0)
        experimentos["knn"].append({"param": k, "f1": round(f1_val * 100, 2)})

    for n in [50, 100, 200, 500]:
        rf_exp = RandomForestClassifier(n_estimators=n, n_jobs=-1, random_state=SEMILLA)
        rf_exp.fit(X_entrenamiento, y_entrenamiento)
        f1_val = f1_score(y_validacion, rf_exp.predict(X_validacion), zero_division=0)
        experimentos["rf"].append({"param": n, "f1": round(f1_val * 100, 2)})

    for d in [2, 3, 5, 8, None]:
        dt_exp = DecisionTreeClassifier(max_depth=d, random_state=SEMILLA)
        dt_exp.fit(X_entrenamiento, y_entrenamiento)
        f1_val = f1_score(y_validacion, dt_exp.predict(X_validacion), zero_division=0)
        experimentos["dt"].append({"param": "Sin límite" if d is None else str(d), "f1": round(f1_val * 100, 2)})

    # --- GRÁFICAS ---
    metricas_lista = ["exactitud", "precision", "recall", "f1_score"]
    plt.figure(figsize=(9, 4.5))
    ancho = 0.20
    posiciones = np.arange(len(df_resultados))
    colores_verdes = ['#a9dfbf', '#7dcea0', '#52be80', '#27ae60']

    for i, metrica in enumerate(metricas_lista):
        desplazamiento = (i - 1.5) * ancho
        plt.bar(posiciones + desplazamiento, df_resultados[metrica], width=ancho, label=metrica.replace("_", " ").title(), color=colores_verdes[i])

    plt.xticks(posiciones, df_resultados["algoritmo"])
    plt.ylim(0, 105)
    plt.ylabel("Resultado (%)")
    plt.xlabel("Algoritmo")
    plt.title("Velox UTVT - Comparacion de algoritmos de clasificacion")
    plt.legend()
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    ruta_grafica_barras = "static/img/ia/p05_comparacion.png"
    plt.savefig(ruta_grafica_barras, dpi=150)
    plt.close()

    matriz_arr = confusion_matrix(y_prueba, predicciones_finales, labels=[0, 1])
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
    ruta_grafica_matriz = "static/img/ia/p05_matriz.png"
    plt.savefig(ruta_grafica_matriz, dpi=150)
    plt.close()

    df_resultados.to_csv("resultados/velox_resultados_comparacion.csv", index=False)

    return templates.TemplateResponse(
        request,
        "ia_modulo/practica05.html",
        {
            "request": request,
            "total_registros": len(df),
            "primeros_datos": df.head(10).to_dict(orient="records"),
            "distribucion_y": df["viaje_exitoso"].value_counts().sort_index().to_dict(),
            "conteos": {"entrenamiento": len(X_entrenamiento), "validacion": len(X_validacion), "prueba": len(X_prueba)},
            "resultados": df_resultados.to_dict(orient="records"),
            "metricas": metricas_finales,
            "reporte_str": reporte_str,
            "matriz": matriz,
            "cv": cv_info,
            "experimentos": experimentos,
            "img_barras": "/" + ruta_grafica_barras,
            "img_matriz": "/" + ruta_grafica_matriz
        }
    )


@router.get("/exportar_csv")
async def exportar_csv():
    ruta_csv = "resultados/velox_resultados_comparacion.csv"
    if os.path.exists(ruta_csv):
        return FileResponse(ruta_csv, media_type="text/csv", filename="velox_resultados_comparacion.csv")
    return {"error": "El archivo CSV aún no ha sido generado."}