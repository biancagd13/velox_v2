# ============================================================
# PROYECTO: VELOX UTVT
# PRACTICA 26 (ETAPA 18): RED NEURONAL MULTICAPA (MLP)
#
# Archivo:
# p26red_neuronal_mlp_velox.py
#
# Objetivo:
# Construir y entrenar una Red Neuronal Multicapa (Multi-Layer
# Perceptron - MLP) con capas ocultas y activación ReLU para
# modelar relaciones no lineales en la viabilidad de viajes de
# Velox UTVT, comparando su desempeño frente a un modelo lineal.
#
# Ejecución desde terminal:
# python p26red_neuronal_mlp_velox.py
# ============================================================

import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import Perceptron
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score
)
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

# Permitir importar 'database.py' desde la raíz del proyecto Velox UTVT
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from database import db


# ============================================================
# 1. CONFIGURACION GENERAL Y DIRECTORIOS
# ============================================================

SEMILLA = 42
ESTADOS_EXITOSOS = ["activo", "aceptado", "en_curso", "finalizado"]

RUTA_ACTUAL = Path(__file__).resolve().parent
CARPETA_DL = RUTA_ACTUAL / "resultados_dl_velox"
CARPETA_DL.mkdir(parents=True, exist_ok=True)


# ============================================================
# 2. CARGA DE DATOS REALES + PATRONES NO LINEALES VELOX UTVT
# ============================================================

def cargar_datos_no_lineales_velox():
    """
    Extrae los viajes reales de 'db.viaje' en MongoDB Atlas y
    complementa hasta 400 registros con fronteras no lineales
    (zona óptima intermedia de tarifa y ocupación) para evaluar
    la capacidad de aprendizaje profundo del MLP.
    """
    print("\n1. CARGA DE DATOS OPERATIVOS CON PATRONES NO LINEALES")
    print("-" * 65)

    registros_reales = []
    try:
        cursor = db.viaje.find({}, {"_id": 0, "costo": 1, "asientos_disponibles": 1, "estado": 1})
        for doc in cursor:
            estado = str(doc.get("estado", "")).lower().strip()
            registros_reales.append({
                "costo": float(doc.get("costo", 15.0)),
                "asientos_disponibles": int(doc.get("asientos_disponibles", 2)),
                "viaje_exitoso": 1 if estado in ESTADOS_EXITOSOS else 0
            })
    except Exception as error:
        print(f"Aviso de conexión MongoDB: {error}")

    df = pd.DataFrame(registros_reales)
    n_reales = len(df)
    print(f"Viajes reales recuperados de MongoDB Atlas: {n_reales}")

    if n_reales < 400:
        rng = np.random.default_rng(SEMILLA)
        n_extra = 400 - n_reales

        costos_sim = rng.uniform(8.0, 42.0, n_extra).round(1)
        asientos_sim = rng.uniform(1.0, 5.0, n_extra).round(1)

        # Frontera no lineal (elíptica / zona dulce operativa):
        # Tarifas entre $12 y $26 MXN con 2 a 4 asientos tienen alta consolidación;
        # tarifas extremas (<$10 o >$30) o baja capacidad tienen menor consolidación.
        dist_costo = ((costos_sim - 18.0) / 8.5) ** 2
        dist_asientos = ((asientos_sim - 3.0) / 1.5) ** 2
        ruido = rng.normal(0, 0.18, n_extra)

        exito_sim = ((dist_costo + dist_asientos + ruido) <= 1.15).astype(int)

        df_sim = pd.DataFrame({
            "costo": costos_sim,
            "asientos_disponibles": asientos_sim,
            "viaje_exitoso": exito_sim
        })
        df = pd.concat([df, df_sim], ignore_index=True)

    print(f"Total de registros preparados: {len(df)}")
    print("Distribución de clases (0=No Exitoso, 1=Exitoso):")
    print(df["viaje_exitoso"].value_counts().sort_index().to_string())

    return df


# ============================================================
# 3. ENTRENAMIENTO COMPARATIVO: PERCEPTRON VS MLP
# ============================================================

def entrenar_modelos_neuronales(X_train_esc, X_test_esc, y_train, y_test):
    """
    Compara un Perceptrón simple (lineal) frente a una Red Neuronal
    Multicapa MLP (2 capas ocultas: 32 y 16 neuronas con ReLU).
    """
    print("\n2. ENTRENAMIENTO: PERCEPTRON LINEAL VS. RED MULTICAPA (MLP)")
    print("-" * 65)

    # Modelo A: Perceptrón lineal (referencia)
    perceptron = Perceptron(max_iter=500, eta0=0.05, random_state=SEMILLA)
    perceptron.fit(X_train_esc, y_train)
    pred_perc = perceptron.predict(X_test_esc)

    # Modelo B: Red Neuronal Multicapa (MLP)
    mlp = MLPClassifier(
        hidden_layer_sizes=(32, 16),
        activation="relu",
        solver="adam",
        alpha=0.001,
        learning_rate_init=0.01,
        max_iter=400,
        early_stopping=True,
        validation_fraction=0.15,
        n_iter_no_change=20,
        random_state=SEMILLA
    )
    mlp.fit(X_train_esc, y_train)
    pred_mlp = mlp.predict(X_test_esc)

    resultados = []
    for nombre, preds in [("Perceptrón Simple (Lineal)", pred_perc), ("Red Neuronal MLP (32, 16)", pred_mlp)]:
        acc = accuracy_score(y_test, preds)
        prec = precision_score(y_test, preds, zero_division=0)
        rec = recall_score(y_test, preds, zero_division=0)
        f1 = f1_score(y_test, preds, zero_division=0)

        print(f"\nModelo: {nombre}")
        print(f"  Accuracy:  {acc * 100:.2f}%")
        print(f"  Precision: {prec * 100:.2f}%")
        print(f"  Recall:    {rec * 100:.2f}%")
        print(f"  F1-Score:  {f1 * 100:.2f}%")

        resultados.append({
            "modelo": nombre,
            "accuracy": round(float(acc), 4),
            "precision": round(float(prec), 4),
            "recall": round(float(rec), 4),
            "f1_score": round(float(f1), 4)
        })

    print(f"\nIteraciones ejecutadas por el MLP: {mlp.n_iter_}")
    print(f"Pérdida final (Loss): {mlp.loss_:.4f}")

    return perceptron, mlp, pred_mlp, pd.DataFrame(resultados)


# ============================================================
# 4. GENERACION DE GRAFICAS (LOSS, FRONTERA NO LINEAL Y MATRIZ)
# ============================================================

def generar_visualizaciones_mlp(mlp, X_todo_esc, y_todo, y_test, pred_mlp):
    """
    Genera y guarda en 'resultados_dl_velox/':
      1. Curva de pérdida (Loss) durante el entrenamiento del MLP.
      2. Frontera de decisión curva no lineal en 2D.
      3. Matriz de confusión del modelo MLP.
    """
    # 1. Curva de pérdida (Loss Curve)
    plt.figure(figsize=(8, 5))
    plt.plot(mlp.loss_curve_, color="#27ae60", linewidth=2, label="Pérdida Entrenamiento (Log-Loss)")
    if hasattr(mlp, "validation_scores_") and mlp.validation_scores_:
        ax2 = plt.gca().twinx()
        ax2.plot(
            [s * 100 for s in mlp.validation_scores_],
            color="#145a32",
            linestyle="--",
            linewidth=1.5,
            label="Accuracy Validación (%)"
        )
        ax2.set_ylabel("Accuracy Validación (%)", color="#145a32")
        ax2.set_ylim(0, 105)

    plt.title("Velox UTVT - Curva de Aprendizaje de la Red Neuronal MLP")
    plt.xlabel("Época / Iteración")
    plt.ylabel("Función de Pérdida (Loss)")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    ruta_loss = CARPETA_DL / "p26_curva_perdida_mlp.png"
    plt.savefig(ruta_loss, dpi=300)
    plt.close()

    # 2. Frontera de decisión no lineal
    plt.figure(figsize=(9, 6))
    x_min, x_max = X_todo_esc[:, 0].min() - 0.5, X_todo_esc[:, 0].max() + 0.5
    y_min, y_max = X_todo_esc[:, 1].min() - 0.5, X_todo_esc[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 250), np.linspace(y_min, y_max, 250))

    malla = np.c_[xx.ravel(), yy.ravel()]
    probs = mlp.predict_proba(malla)[:, 1].reshape(xx.shape)

    contorno = plt.contourf(xx, yy, probs, levels=20, cmap="Greens", alpha=0.75)
    plt.colorbar(contorno, label="Probabilidad de Consolidación (MLP)")
    plt.contour(xx, yy, probs, levels=[0.5], colors=["#145a32"], linewidths=2, linestyles="--")

    clase_0 = X_todo_esc[y_todo == 0]
    clase_1 = X_todo_esc[y_todo == 1]

    plt.scatter(clase_0[:, 0], clase_0[:, 1], color="#e74c3c", label="No Exitoso (0)", edgecolors="black", s=35, alpha=0.85)
    plt.scatter(clase_1[:, 0], clase_1[:, 1], color="#27ae60", label="Viaje Exitoso (1)", edgecolors="black", s=35, alpha=0.85)

    plt.title("Velox UTVT - Frontera de Decisión No Lineal (MLP 32x16)")
    plt.xlabel("Costo Estandarizado (z-score)")
    plt.ylabel("Asientos Disponibles Estandarizados (z-score)")
    plt.legend()
    plt.grid(alpha=0.25)
    plt.tight_layout()
    ruta_frontera = CARPETA_DL / "p26_frontera_no_lineal_mlp.png"
    plt.savefig(ruta_frontera, dpi=300)
    plt.close()

    # 3. Matriz de confusión
    matriz = confusion_matrix(y_test, pred_mlp, labels=[0, 1])
    disp = ConfusionMatrixDisplay(confusion_matrix=matriz, display_labels=["No Exitoso", "Exitoso"])
    disp.plot(cmap="Greens", values_format="d")
    plt.title("Matriz de Confusión - Red Neuronal MLP (Velox UTVT)")
    plt.tight_layout()
    ruta_matriz = CARPETA_DL / "p26_matriz_confusion_mlp.png"
    plt.savefig(ruta_matriz, dpi=300)
    plt.close()

    return ruta_loss, ruta_frontera, ruta_matriz


# ============================================================
# 5. PUNTO DE ENTRADA
# ============================================================

def main():
    print("=" * 65)
    print("VELOX UTVT - PRACTICA 26 (ETAPA 18)")
    print("RED NEURONAL MULTICAPA (MLP) PARA PATRONES NO LINEALES")
    print("=" * 65)

    df = cargar_datos_no_lineales_velox()

    X = df[["costo", "asientos_disponibles"]].to_numpy(dtype=float)
    y = df["viaje_exitoso"].to_numpy(dtype=int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=SEMILLA, stratify=y
    )

    escalador = StandardScaler()
    X_train_esc = escalador.fit_transform(X_train)
    X_test_esc = escalador.transform(X_test)

    _, mlp, pred_mlp, df_comparativa = entrenar_modelos_neuronales(
        X_train_esc, X_test_esc, y_train, y_test
    )

    print("\n3. REPORTE DE CLASIFICACION DETALLADO (MLP)")
    print("-" * 65)
    print(
        classification_report(
            y_test,
            pred_mlp,
            labels=[0, 1],
            target_names=["No Exitoso", "Exitoso"],
            zero_division=0
        )
    )

    ruta_csv = CARPETA_DL / "p26_comparativa_mlp.csv"
    df_comparativa.to_csv(ruta_csv, index=False, encoding="utf-8-sig")

    X_todo_esc = escalador.transform(X)
    ruta_loss, ruta_front, ruta_mat = generar_visualizaciones_mlp(
        mlp, X_todo_esc, y, y_test, pred_mlp
    )

    print("\n4. ARCHIVOS GENERADOS EN 'resultados_dl_velox/'")
    print("-" * 65)
    print(f"- Tabla comparativa CSV:  {ruta_csv}")
    print(f"- Curva de pérdida:       {ruta_loss}")
    print(f"- Frontera no lineal 2D:  {ruta_front}")
    print(f"- Matriz de confusión:    {ruta_mat}")
    print("\nPRACTICA 26 FINALIZADA CORRECTAMENTE.")


if __name__ == "__main__":
    main()