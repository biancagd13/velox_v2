# ============================================================
# PROYECTO: VELOX UTVT
# PRACTICA 25 (ETAPA 17): NEURONA ARTIFICIAL Y PERCEPTRON
#
# Archivo:
# p25neurona_perceptron_velox.py
#
# Objetivo:
# Implementar desde cero una neurona artificial (Perceptrón de
# Rosenblatt) utilizando únicamente NumPy para clasificar la
# viabilidad de viajes en Velox UTVT a partir de sus variables
# operativas (costo y asientos_disponibles), visualizando el
# aprendizaje época por época y su frontera de decisión lineal.
#
# Ejecución desde terminal:
# python p25neurona_perceptron_velox.py
# ============================================================

import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# Permitir importar 'database.py' desde la raíz del proyecto Velox UTVT
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from database import db


# ============================================================
# 1. CONFIGURACION GENERAL Y DIRECTORIOS
# ============================================================

SEMILLA = 42
TASA_APRENDIZAJE = 0.05
EPOCAS = 60
ESTADOS_EXITOSOS = ["activo", "aceptado", "en_curso", "finalizado"]

RUTA_ACTUAL = Path(__file__).resolve().parent
CARPETA_DL = RUTA_ACTUAL / "resultados_dl_velox"
CARPETA_DL.mkdir(parents=True, exist_ok=True)


# ============================================================
# 2. CARGA DE DATOS REALES DE MONGODB + CALIBRACION
# ============================================================

def cargar_datos_velox():
    """
    Extrae los viajes reales de 'db.viaje' en MongoDB Atlas y
    complementa hasta 200 muestras calibradas al dominio Velox UTVT
    para entrenar el Perceptrón con estabilidad numérica.
    """
    print("\n1. CARGA DE DATOS OPERATIVOS (VELOX UTVT)")
    print("-" * 65)

    registros_reales = []
    try:
        cursor = db.viaje.find({}, {"_id": 0, "costo": 1, "asientos_disponibles": 1, "estado": 1})
        for doc in cursor:
            estado = str(doc.get("estado", "")).lower().strip()
            registros_reales.append({
                "costo": float(doc.get("costo", 15.0)),
                "asientos_disponibles": int(doc.get("asientos_disponibles", 2)),
                "viaje_exitoso": 1 if estado in ESTADOS_EXITOSOS else 0,
                "origen_dato": "MongoDB_Real"
            })
    except Exception as error:
        print(f"Aviso de conexión MongoDB: {error}")

    df = pd.DataFrame(registros_reales)
    n_reales = len(df)
    print(f"Viajes reales recuperados de MongoDB Atlas: {n_reales}")

    if n_reales < 200:
        rng = np.random.default_rng(SEMILLA)
        n_extra = 200 - n_reales
        costos_sim = rng.uniform(10.0, 38.0, n_extra).round(1)
        asientos_sim = rng.integers(1, 5, n_extra)

        # Frontera lineal separable con ligero margen real (costo <= 22 y asientos >= 2)
        score = (23.0 - costos_sim) * 0.45 + (asientos_sim - 1.8) * 1.6 + rng.normal(0, 0.35, n_extra)
        exito_sim = (score >= 0).astype(int)

        df_sim = pd.DataFrame({
            "costo": costos_sim,
            "asientos_disponibles": asientos_sim,
            "viaje_exitoso": exito_sim,
            "origen_dato": "Calibracion_UTVT"
        })
        df = pd.concat([df, df_sim], ignore_index=True)

    print(f"Total de registros para entrenamiento/prueba: {len(df)}")
    print("Distribución de clases (0=No Exitoso, 1=Exitoso):")
    print(df["viaje_exitoso"].value_counts().sort_index().to_string())

    return df


# ============================================================
# 3. IMPLEMENTACION DE LA NEURONA ARTIFICIAL (PERCEPTRON)
# ============================================================

class PerceptronVelox:
    """
    Neurona artificial binaria implementada desde cero con NumPy.
    Calcula:
        z = w1*x1 + w2*x2 + b
        y_hat = 1 si z >= 0, en otro caso 0
    Actualiza pesos mediante la regla de Rosenblatt:
        w = w + eta * (y - y_hat) * x
        b = b + eta * (y - y_hat)
    """

    def __init__(self, tasa_aprendizaje=0.05, epocas=60, semilla=42):
        self.eta = tasa_aprendizaje
        self.epocas = epocas
        self.semilla = semilla
        self.pesos = None
        self.sesgo = 0.0
        self.historial_errores = []
        self.historial_accuracy = []

    def funcion_activacion_escalon(self, z):
        return np.where(z >= 0.0, 1, 0)

    def funcion_sigmoide(self, z):
        z_clip = np.clip(z, -250, 250)
        return 1.0 / (1.0 + np.exp(-z_clip))

    def suma_ponderada(self, X):
        return np.dot(X, self.pesos) + self.sesgo

    def predecir(self, X):
        z = self.suma_ponderada(X)
        return self.funcion_activacion_escalon(z)

    def predecir_probabilidad(self, X):
        z = self.suma_ponderada(X)
        return self.funcion_sigmoide(z)

    def entrenar(self, X, y):
        rng = np.random.default_rng(self.semilla)
        n_muestras, n_caracteristicas = X.shape

        self.pesos = rng.normal(loc=0.0, scale=0.01, size=n_caracteristicas)
        self.sesgo = 0.0
        self.historial_errores = []
        self.historial_accuracy = []

        for epoca in range(1, self.epocas + 1):
            errores_epoca = 0

            # Barajar índices en cada época
            indices = rng.permutation(n_muestras)
            for idx in indices:
                xi = X[idx]
                yi = y[idx]

                z_i = np.dot(xi, self.pesos) + self.sesgo
                y_hat = 1 if z_i >= 0.0 else 0

                error = yi - y_hat
                if error != 0:
                    self.pesos += self.eta * error * xi
                    self.sesgo += self.eta * error
                    errores_epoca += 1

            preds_epoca = self.predecir(X)
            acc_epoca = accuracy_score(y, preds_epoca)

            self.historial_errores.append(errores_epoca)
            self.historial_accuracy.append(acc_epoca)

            if epoca == 1 or epoca % 10 == 0 or errores_epoca == 0:
                print(
                    f"  Época {epoca:02d}/{self.epocas:02d} | "
                    f"Errores: {errores_epoca:03d} | "
                    f"Accuracy entrenamiento: {acc_epoca * 100:.2f}%"
                )

            if errores_epoca == 0:
                print(f"  -> Convergencia exacta alcanzada en la época {epoca}.")
                break


# ============================================================
# 4. VISUALIZACIONES (CONVERGENCIA Y FRONTERA DE DECISION)
# ============================================================

def generar_graficas_perceptron(perceptron, X_escalado, y, escalador):
    """
    Genera:
      1. Gráfica de evolución de errores y precisión por época.
      2. Frontera de decisión lineal aprendida en el plano Costo vs. Asientos.
    """
    # Gráfica 1: Convergencia del aprendizaje
    epocas_eje = range(1, len(perceptron.historial_errores) + 1)

    fig, ejes = plt.subplots(1, 2, figsize=(12, 5))

    ejes[0].plot(epocas_eje, perceptron.historial_errores, marker="o", color="#e74c3c", linewidth=2)
    ejes[0].set_title("Errores de Clasificación por Época")
    ejes[0].set_xlabel("Época")
    ejes[0].set_ylabel("Número de Errores")
    ejes[0].grid(alpha=0.3)

    ejes[1].plot(epocas_eje, [a * 100 for a in perceptron.historial_accuracy], marker="s", color="#27ae60", linewidth=2)
    ejes[1].set_title("Evolución de Accuracy en Entrenamiento (%)")
    ejes[1].set_xlabel("Época")
    ejes[1].set_ylabel("Accuracy (%)")
    ejes[1].set_ylim(0, 105)
    ejes[1].grid(alpha=0.3)

    plt.suptitle("Velox UTVT - Aprendizaje de la Neurona Artificial (Perceptrón)", fontsize=12, fontweight="bold")
    plt.tight_layout()
    ruta_conv = CARPETA_DL / "p25_convergencia_perceptron.png"
    plt.savefig(ruta_conv, dpi=300)
    plt.close()

    # Gráfica 2: Frontera de decisión lineal en espacio estandarizado
    plt.figure(figsize=(9, 6))

    x_min, x_max = X_escalado[:, 0].min() - 0.5, X_escalado[:, 0].max() + 0.5
    y_min, y_max = X_escalado[:, 1].min() - 0.5, X_escalado[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 250), np.linspace(y_min, y_max, 250))

    malla = np.c_[xx.ravel(), yy.ravel()]
    Z = perceptron.predecir(malla).reshape(xx.shape)

    plt.contourf(xx, yy, Z, alpha=0.25, cmap="Greens")

    clase_0 = X_escalado[y == 0]
    clase_1 = X_escalado[y == 1]

    plt.scatter(clase_0[:, 0], clase_0[:, 1], color="#e74c3c", label="No Exitoso (0)", edgecolors="black", alpha=0.8)
    plt.scatter(clase_1[:, 0], clase_1[:, 1], color="#27ae60", label="Viaje Exitoso (1)", edgecolors="black", alpha=0.8)

    # Recta de frontera w1*x1 + w2*x2 + b = 0
    w1, w2 = perceptron.pesos
    b = perceptron.sesgo
    if abs(w2) > 1e-6:
        x_vals = np.linspace(x_min, x_max, 100)
        y_vals = -(w1 * x_vals + b) / w2
        plt.plot(x_vals, y_vals, color="#145a32", linestyle="--", linewidth=2, label="Frontera Lineal (z=0)")

    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    plt.xlabel("Costo Estandarizado (z-score)")
    plt.ylabel("Asientos Disponibles Estandarizados (z-score)")
    plt.title("Velox UTVT - Frontera de Decisión del Perceptrón")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    ruta_frontera = CARPETA_DL / "p25_frontera_decision.png"
    plt.savefig(ruta_frontera, dpi=300)
    plt.close()

    return ruta_conv, ruta_frontera


# ============================================================
# 5. PUNTO DE ENTRADA
# ============================================================

def main():
    print("=" * 65)
    print("VELOX UTVT - PRACTICA 25 (ETAPA 17)")
    print("NEURONA ARTIFICIAL Y PERCEPTRON DESDE CERO")
    print("=" * 65)

    df = cargar_datos_velox()

    X = df[["costo", "asientos_disponibles"]].to_numpy(dtype=float)
    y = df["viaje_exitoso"].to_numpy(dtype=int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=SEMILLA, stratify=y
    )

    escalador = StandardScaler()
    X_train_esc = escalador.fit_transform(X_train)
    X_test_esc = escalador.transform(X_test)

    print("\n2. ENTRENAMIENTO DE LA NEURONA ARTIFICIAL (ROSENBLATT)")
    print("-" * 65)

    neurona = PerceptronVelox(
        tasa_aprendizaje=TASA_APRENDIZAJE,
        epocas=EPOCAS,
        semilla=SEMILLA
    )
    neurona.entrenar(X_train_esc, y_train)

    print("\n3. PARAMETROS SINAPTICOS APRENDIDOS")
    print("-" * 65)
    print(f"Peso w1 (costo):                {neurona.pesos[0]:.4f}")
    print(f"Peso w2 (asientos_disponibles): {neurona.pesos[1]:.4f}")
    print(f"Sesgo (bias b):                 {neurona.sesgo:.4f}")

    # Evaluación en conjunto de prueba
    predicciones = neurona.predecir(X_test_esc)
    acc = accuracy_score(y_test, predicciones)
    prec = precision_score(y_test, predicciones, zero_division=0)
    rec = recall_score(y_test, predicciones, zero_division=0)
    f1 = f1_score(y_test, predicciones, zero_division=0)
    matriz = confusion_matrix(y_test, predicciones)

    print("\n4. EVALUACION EN CONJUNTO DE PRUEBA (20%)")
    print("-" * 65)
    print(f"Accuracy:  {acc * 100:.2f}%")
    print(f"Precision: {prec * 100:.2f}%")
    print(f"Recall:    {rec * 100:.2f}%")
    print(f"F1-Score:  {f1 * 100:.2f}%")
    print("Matriz de Confusión:")
    print(matriz)

    # Exportar pesos y métricas
    df_pesos = pd.DataFrame([{
        "peso_w1_costo": round(float(neurona.pesos[0]), 6),
        "peso_w2_asientos": round(float(neurona.pesos[1]), 6),
        "sesgo_b": round(float(neurona.sesgo), 6),
        "accuracy_test": round(float(acc), 4),
        "precision_test": round(float(prec), 4),
        "recall_test": round(float(rec), 4),
        "f1_test": round(float(f1), 4)
    }])
    ruta_csv = CARPETA_DL / "p25_pesos_perceptron.csv"
    df_pesos.to_csv(ruta_csv, index=False, encoding="utf-8-sig")

    X_todo_esc = escalador.transform(X)
    ruta_conv, ruta_front = generar_graficas_perceptron(neurona, X_todo_esc, y, escalador)

    print("\n5. ARCHIVOS GENERADOS EN 'resultados_dl_velox/'")
    print("-" * 65)
    print(f"- Pesos y métricas CSV: {ruta_csv}")
    print(f"- Gráfica convergencia: {ruta_conv}")
    print(f"- Gráfica frontera:     {ruta_front}")
    print("\nPRACTICA 25 FINALIZADA CORRECTAMENTE.")


if __name__ == "__main__":
    main()