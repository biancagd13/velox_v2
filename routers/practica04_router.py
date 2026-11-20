import os
import matplotlib
matplotlib.use('Agg')  # Evita abrir ventanas emergentes en el servidor web

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

from database import db

router = APIRouter(prefix="/ia/practica04", tags=["IA - Práctica 04"])
templates = Jinja2Templates(directory="templates")

# Asegurar que exista la carpeta para guardar las gráficas estáticas
os.makedirs("static/img/ia", exist_ok=True)
SEMILLA = 42


def obtener_datos_mongodb():
    lista_viajes = []
    cursor = db.viaje.find({}, {"_id": 0, "costo": 1, "asientos_disponibles": 1, "estado": 1})
    for doc in cursor:
        costo = float(doc.get("costo", 0))
        asientos = int(doc.get("asientos_disponibles", 0))
        estado = doc.get("estado", "")
        exitoso = 1 if estado in ["en_curso", "finalizado", "activo"] else 0
        lista_viajes.append({
            "costo": costo,
            "asientos_disponibles": asientos,
            "viaje_exitoso": exitoso
        })
    return pd.DataFrame(lista_viajes) if lista_viajes else pd.DataFrame(columns=["costo", "asientos_disponibles", "viaje_exitoso"])


@router.get("/", response_class=HTMLResponse)
async def vista_practica04(request: Request):
    df = obtener_datos_mongodb()

    if len(df) < 5:
        return templates.TemplateResponse("practica04.html", {
            "request": request,
            "error": "No hay suficientes registros en MongoDB (mínimo 5 viajes) para ejecutar el entrenamiento."
        })

    X = df[["costo", "asientos_disponibles"]]
    y = df["viaje_exitoso"]

    # División 60% Entrenamiento, 20% Validación, 20% Prueba
    if len(df) >= 10 and y.value_counts().min() >= 3:
        X_ent, X_temp, y_ent, y_temp = train_test_split(X, y, test_size=0.40, random_state=SEMILLA, stratify=y)
        X_val, X_pru, y_val, y_pru = train_test_split(X_temp, y_temp, test_size=0.50, random_state=SEMILLA, stratify=y_temp)
    else:
        X_ent, X_temp, y_ent, y_temp = train_test_split(X, y, test_size=0.40, random_state=SEMILLA)
        X_val, X_pru, y_val, y_pru = train_test_split(X_temp, y_temp, test_size=0.50, random_state=SEMILLA)

    profundidades = [1, 2, 3, 5, 10, None]
    resultados = []
    modelos = {}

    for prof in profundidades:
        modelo = DecisionTreeClassifier(max_depth=prof, random_state=SEMILLA)
        modelo.fit(X_ent, y_ent)

        acc_ent = accuracy_score(y_ent, modelo.predict(X_ent))
        acc_val = accuracy_score(y_val, modelo.predict(X_val))
        diferencia = acc_ent - acc_val

        if acc_ent < 0.60 and acc_val < 0.60:
            diagnostico = "Subajuste (Underfitting)"
            badge = "warning"
        elif diferencia > 0.15:
            diagnostico = "Sobreajuste (Overfitting)"
            badge = "danger"
        else:
            diagnostico = "Buen Ajuste"
            badge = "success"

        modelos[prof] = modelo
        resultados.append({
            "profundidad": "Sin límite" if prof is None else str(prof),
            "profundidad_grafica": 12 if prof is None else prof,
            "entrenamiento": round(acc_ent * 100, 2),
            "validacion": round(acc_val * 100, 2),
            "diferencia": round(diferencia * 100, 2),
            "diagnostico": diagnostico,
            "badge": badge
        })

    df_res = pd.DataFrame(resultados)

    # Selección del mejor modelo por exactitud de validación
    idx_mejor = df_res["validacion"].idxmax()
    mejor_fila = df_res.loc[idx_mejor]
    prof_sel = None if int(mejor_fila["profundidad_grafica"]) == 12 else int(mejor_fila["profundidad_grafica"])
    mejor_modelo = modelos[prof_sel]

    # Evaluación final con conjunto de Prueba
    pred_prueba = mejor_modelo.predict(X_pru)
    metricas_finales = {
        "profundidad": mejor_fila["profundidad"],
        "exactitud": round(accuracy_score(y_pru, pred_prueba) * 100, 2),
        "precision": round(precision_score(y_pru, pred_prueba, zero_division=0) * 100, 2),
        "recall": round(recall_score(y_pru, pred_prueba, zero_division=0) * 100, 2),
        "f1": round(f1_score(y_pru, pred_prueba, zero_division=0) * 100, 2),
    }

    # Validación cruzada
    cv_folds = min(5, len(X_ent)) if len(X_ent) >= 2 else 2
    scores_cv = cross_val_score(
        DecisionTreeClassifier(max_depth=prof_sel, random_state=SEMILLA),
        X_ent, y_ent, cv=cv_folds, scoring="accuracy"
    )
    cv_info = {
        "promedio": round(scores_cv.mean() * 100, 2),
        "desviacion": round(scores_cv.std() * 100, 2)
    }

    # --- GRÁFICA 1: ENTRENAMIENTO VS VALIDACIÓN (Tonos Verdes + Offset Visual) ---
    plt.figure(figsize=(9, 4.5))
    x_vals = df_res["profundidad_grafica"]
    offset = 0.8  # Desfase en escala 0-100%

    plt.plot(x_vals, df_res["entrenamiento"] + offset, marker="o", linewidth=2.5, linestyle="-", color="#1b5e20", label="Entrenamiento")
    plt.plot(x_vals, df_res["validacion"] - offset, marker="s", linewidth=2.5, linestyle="--", color="#66bb6a", label="Validación")

    plt.xticks(x_vals, df_res["profundidad"])
    plt.title("Velox UTVT - Curva de Aprendizaje (Entrenamiento vs Validación)")
    plt.xlabel("Profundidad del Árbol de Decisión")
    plt.ylabel("Exactitud (%)")
    plt.ylim(0, 108)
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.legend()
    plt.tight_layout()
    ruta_grafica_curva = "static/img/ia/p04_comparacion.png"
    plt.savefig(ruta_grafica_curva, dpi=150)
    plt.close()

    # --- GRÁFICA 2: MATRIZ DE CONFUSIÓN (Paleta Greens) ---
    matriz = confusion_matrix(y_pru, pred_prueba, labels=[0, 1])
    plt.figure(figsize=(5, 4.5))
    plt.imshow(matriz, cmap="Greens")
    plt.title(f"Matriz de Confusión (Profundidad: {mejor_fila['profundidad']})")
    plt.xlabel("Predicción del Modelo")
    plt.ylabel("Resultado Real")
    plt.xticks([0, 1], ["No exitoso", "Exitoso"])
    plt.yticks([0, 1], ["No exitoso", "Exitoso"])

    for f in range(2):
        for c in range(2):
            color_texto = "white" if matriz[f, c] > matriz.max() / 2 else "black"
            plt.text(c, f, str(matriz[f, c]), ha="center", va="center", fontsize=14, color=color_texto, fontweight="bold")

    plt.colorbar()
    plt.tight_layout()
    ruta_grafica_matriz = "static/img/ia/p04_matriz.png"
    plt.savefig(ruta_grafica_matriz, dpi=150)
    plt.close()

    return templates.TemplateResponse(
        request, 
        "ia_modulo/practica04.html", 
        {
            "total_registros": len(df),
            "conteos": {
                "entrenamiento": len(X_ent),
                "validacion": len(X_val),
                "prueba": len(X_pru)
            },
            "resultados": resultados,
            "metricas": metricas_finales,
            "cv": cv_info,
            "img_curva": "/" + ruta_grafica_curva,
            "img_matriz": "/" + ruta_grafica_matriz
        }
    )