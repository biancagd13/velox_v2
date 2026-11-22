# ============================================================
# PROYECTO: VELOX UTVT
# PRACTICA 27 (ETAPA 19): DEEP LEARNING CON PYTORCH
#
# Archivo:
# p27deep_learning_pytorch_velox.py
#
# Objetivo:
# Construir, entrenar, evaluar y serializar una Red Neuronal
# Profunda (Deep Learning) en PyTorch utilizando tensores,
# DataLoader por mini-lotes, BatchNorm, Dropout y optimizador
# Adam para predecir la viabilidad operativa de viajes en
# Velox UTVT, ejecutando inferencia en producción desde el
# modelo guardado (.pth).
#
# Ejecución desde terminal:
# python p27deep_learning_pytorch_velox.py
# ============================================================

import os
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# Permitir importar 'database.py' desde la raíz del proyecto Velox UTVT
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from database import db


# ============================================================
# 1. CONFIGURACION DE ENTORNO, SEMILLAS Y RUTAS
# ============================================================

SEMILLA = 42
EPOCAS = 35
TAMANO_LOTE = 64
TASA_APRENDIZAJE = 0.001
ESTADOS_EXITOSOS = ["activo", "aceptado", "en_curso", "finalizado"]

np.random.seed(SEMILLA)
torch.manual_seed(SEMILLA)

DISPOSITIVO = torch.device("cuda" if torch.cuda.is_available() else "cpu")

RUTA_ACTUAL = Path(__file__).resolve().parent
CARPETA_DL = RUTA_ACTUAL / "resultados_dl_velox"
CARPETA_DL.mkdir(parents=True, exist_ok=True)
CARPETA_BIGDATA = RUTA_ACTUAL / "resultados_bigdata_velox"

ARCHIVO_MODELO_PTH = CARPETA_DL / "modelo_velox_pytorch.pth"
REPORTE_METRICAS_CSV = CARPETA_DL / "p27_metricas_pytorch.csv"
GRAFICA_CURVAS = CARPETA_DL / "p27_curvas_entrenamiento_pytorch.png"
GRAFICA_MATRIZ = CARPETA_DL / "p27_matriz_confusion_pytorch.png"

COLUMNAS_CARACTERISTICAS = [
    "distancia_km",
    "costo",
    "asientos_disponibles",
    "puntualidad",
    "calificacion_conductor"
]


# ============================================================
# 2. CARGA INTEGRAL DE DATOS (MONGODB ATLAS + BIG DATA)
# ============================================================

def cargar_dataset_profundo():
    """
    Combina los registros reales de MongoDB Atlas ('db.viaje') con
    el dataset masivo de Velox UTVT (si existe de P20/P21) o genera
    una muestra calibrada de 5,000 registros para entrenar en PyTorch.
    """
    print("\n1. PREPARACION DE DATASET PARA DEEP LEARNING (PYTORCH)")
    print("-" * 65)

    registros_reales = []
    try:
        cursor = db.viaje.find({}, {"_id": 0, "costo": 1, "asientos_disponibles": 1, "estado": 1})
        for doc in cursor:
            estado = str(doc.get("estado", "")).lower().strip()
            registros_reales.append({
                "distancia_km": 18.0,
                "costo": float(doc.get("costo", 15.0)),
                "asientos_disponibles": int(doc.get("asientos_disponibles", 2)),
                "puntualidad": 90.0,
                "calificacion_conductor": 4.8,
                "viaje_exitoso": 1 if estado in ESTADOS_EXITOSOS else 0
            })
    except Exception as error:
        print(f"Aviso de conexión MongoDB Atlas: {error}")

    df_real = pd.DataFrame(registros_reales)
    print(f"Registros reales recuperados de MongoDB Atlas: {len(df_real)}")

    # Verificar si ya existe el dataset masivo generado en P20 / P21
    ruta_pq = CARPETA_BIGDATA / "velox_viajes_masivos_50k.parquet"
    ruta_csv = CARPETA_BIGDATA / "velox_viajes_masivos_50k.csv"

    if ruta_pq.exists():
        df_masivo = pd.read_parquet(ruta_pq, columns=COLUMNAS_CARACTERISTICAS + ["viaje_exitoso"])
        df_masivo = df_masivo.sample(n=min(10_000, len(df_masivo)), random_state=SEMILLA)
        print(f"Fuente complementaria cargada desde Parquet (P21): {len(df_masivo):,} registros")
    elif ruta_csv.exists():
        df_masivo = pd.read_csv(ruta_csv, usecols=COLUMNAS_CARACTERISTICAS + ["viaje_exitoso"])
        df_masivo = df_masivo.sample(n=min(10_000, len(df_masivo)), random_state=SEMILLA)
        print(f"Fuente complementaria cargada desde CSV (P20): {len(df_masivo):,} registros")
    else:
        rng = np.random.default_rng(SEMILLA)
        n_sim = 5_000
        distancias = rng.choice([9.5, 13.0, 15.0, 16.5, 19.0, 22.5, 24.0], size=n_sim)
        costos = np.clip(rng.normal(17.5, 6.0, size=n_sim), 8.0, 50.0).round(2)
        asientos = rng.integers(1, 6, size=n_sim)
        puntualidad = np.clip(rng.normal(86.0, 10.0, size=n_sim), 40.0, 100.0).round(1)
        calif = np.clip(rng.normal(4.5, 0.45, size=n_sim), 2.5, 5.0).round(2)

        logit = (
            0.45
            + 1.35 * ((30.0 - costos) / 20.0)
            + 0.55 * ((asientos - 2.0) / 3.0)
            + 0.90 * ((puntualidad - 75.0) / 25.0)
            + 0.75 * ((calif - 3.8) / 1.2)
            + rng.normal(0, 0.30, size=n_sim)
        )
        exito = (1.0 / (1.0 + np.exp(-logit)) >= 0.52).astype(int)

        df_masivo = pd.DataFrame({
            "distancia_km": distancias,
            "costo": costos,
            "asientos_disponibles": asientos,
            "puntualidad": puntualidad,
            "calificacion_conductor": calif,
            "viaje_exitoso": exito
        })
        print(f"Dataset calibrado generado en memoria: {len(df_masivo):,} registros")

    df_total = pd.concat([df_real, df_masivo], ignore_index=True)
    print(f"Total de muestras consolidadas: {len(df_total):,}")
    return df_total


# ============================================================
# 3. ARQUITECTURA DE RED NEURONAL PROFUNDA EN PYTORCH
# ============================================================

class RedNeuronalVeloxUTVT(nn.Module):
    """
    Arquitectura Deep Learning tabular para Velox UTVT:
      Entrada (5 features) -> Linear(64) + BatchNorm + ReLU + Dropout(0.20)
                           -> Linear(32) + BatchNorm + ReLU + Dropout(0.15)
                           -> Linear(16) + ReLU
                           -> Salida Logit (1)
    """

    def __init__(self, num_entradas):
        super().__init__()
        self.red = nn.Sequential(
            nn.Linear(num_entradas, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.20),

            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(0.15),

            nn.Linear(32, 16),
            nn.ReLU(),

            nn.Linear(16, 1)
        )

    def forward(self, x):
        return self.red(x)


# ============================================================
# 4. ENTRENAMIENTO Y VALIDACION POR MINI-LOTES
# ============================================================

def entrenar_modelo_pytorch(X_train_esc, y_train, X_val_esc, y_val):
    """
    Entrena la red neuronal profunda usando TensorDataset, DataLoader,
    BCEWithLogitsLoss y optimizador Adam.
    """
    print(f"\n2. ENTRENAMIENTO EN PYTORCH (Dispositivo: {DISPOSITIVO})")
    print("-" * 65)

    train_dataset = TensorDataset(
        torch.tensor(X_train_esc, dtype=torch.float32),
        torch.tensor(y_train.reshape(-1, 1), dtype=torch.float32)
    )
    val_dataset = TensorDataset(
        torch.tensor(X_val_esc, dtype=torch.float32),
        torch.tensor(y_val.reshape(-1, 1), dtype=torch.float32)
    )

    train_loader = DataLoader(train_dataset, batch_size=TAMANO_LOTE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=TAMANO_LOTE, shuffle=False)

    modelo = RedNeuronalVeloxUTVT(num_entradas=X_train_esc.shape[1]).to(DISPOSITIVO)
    criterio = nn.BCEWithLogitsLoss()
    optimizador = torch.optim.Adam(modelo.parameters(), lr=TASA_APRENDIZAJE, weight_decay=1e-4)

    historial = {
        "loss_train": [],
        "loss_val": [],
        "acc_train": [],
        "acc_val": []
    }

    inicio = time.perf_counter()

    for epoca in range(1, EPOCAS + 1):
        # Fase de entrenamiento
        modelo.train()
        perdida_acum_train = 0.0
        aciertos_train = 0
        total_train = 0

        for bx, by in train_loader:
            bx, by = bx.to(DISPOSITIVO), by.to(DISPOSITIVO)
            optimizador.zero_grad()

            logits = modelo(bx)
            loss = criterio(logits, by)
            loss.backward()
            optimizador.step()

            perdida_acum_train += loss.item() * bx.size(0)
            preds = (torch.sigmoid(logits) >= 0.5).float()
            aciertos_train += (preds == by).sum().item()
            total_train += bx.size(0)

        # Fase de validación
        modelo.eval()
        perdida_acum_val = 0.0
        aciertos_val = 0
        total_val = 0

        with torch.no_grad():
            for bx, by in val_loader:
                bx, by = bx.to(DISPOSITIVO), by.to(DISPOSITIVO)
                logits = modelo(bx)
                loss = criterio(logits, by)

                perdida_acum_val += loss.item() * bx.size(0)
                preds = (torch.sigmoid(logits) >= 0.5).float()
                aciertos_val += (preds == by).sum().item()
                total_val += bx.size(0)

        loss_tr = perdida_acum_train / total_train
        loss_vl = perdida_acum_val / total_val
        acc_tr = aciertos_train / total_train
        acc_vl = aciertos_val / total_val

        historial["loss_train"].append(loss_tr)
        historial["loss_val"].append(loss_vl)
        historial["acc_train"].append(acc_tr)
        historial["acc_val"].append(acc_vl)

        if epoca == 1 or epoca % 5 == 0 or epoca == EPOCAS:
            print(
                f"  Época {epoca:02d}/{EPOCAS:02d} | "
                f"Loss Train: {loss_tr:.4f} | Loss Val: {loss_vl:.4f} | "
                f"Acc Val: {acc_vl * 100:.2f}%"
            )

    duracion = time.perf_counter() - inicio
    print(f"\nEntrenamiento completado en {duracion:.2f} segundos.")
    return modelo, historial


# ============================================================
# 5. VISUALIZACIONES EN TONOS VERDES VELOX UTVT
# ============================================================

def generar_graficas_pytorch(historial, y_test, preds_test):
    """Genera las curvas de pérdida/precisión y matriz de confusión."""
    epocas_eje = range(1, len(historial["loss_train"]) + 1)

    fig, ejes = plt.subplots(1, 2, figsize=(12, 5))

    ejes[0].plot(epocas_eje, historial["loss_train"], label="Pérdida Entrenamiento", color="#27ae60", linewidth=2)
    ejes[0].plot(epocas_eje, historial["loss_val"], label="Pérdida Validación", color="#145a32", linestyle="--", linewidth=2)
    ejes[0].set_title("Evolución de Función de Pérdida (BCEWithLogitsLoss)")
    ejes[0].set_xlabel("Época")
    ejes[0].set_ylabel("Loss")
    ejes[0].legend()
    ejes[0].grid(alpha=0.3)

    ejes[1].plot(epocas_eje, [a * 100 for a in historial["acc_train"]], label="Accuracy Entrenamiento", color="#52be80", linewidth=2)
    ejes[1].plot(epocas_eje, [a * 100 for a in historial["acc_val"]], label="Accuracy Validación", color="#145a32", linestyle="--", linewidth=2)
    ejes[1].set_title("Evolución de Accuracy (%)")
    ejes[1].set_xlabel("Época")
    ejes[1].set_ylabel("Accuracy (%)")
    ejes[1].set_ylim(50, 100)
    ejes[1].legend()
    ejes[1].grid(alpha=0.3)

    plt.suptitle("Velox UTVT - Entrenamiento Deep Learning (PyTorch)", fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig(GRAFICA_CURVAS, dpi=300)
    plt.close()

    matriz = confusion_matrix(y_test, preds_test, labels=[0, 1])
    disp = ConfusionMatrixDisplay(confusion_matrix=matriz, display_labels=["No Exitoso", "Exitoso"])
    disp.plot(cmap="Greens", values_format="d")
    plt.title("Matriz de Confusión - Red Neuronal PyTorch (Velox UTVT)")
    plt.tight_layout()
    plt.savefig(GRAFICA_MATRIZ, dpi=300)
    plt.close()


# ============================================================
# 6. SERIALIZACION (.PTH) E INFERENCIA EN PRODUCCION
# ============================================================

def guardar_y_probar_inferencia(modelo, escalador):
    """
    Guarda el checkpoint completo de PyTorch en disco, lo recarga
    y ejecuta predicciones sobre escenarios reales de Velox UTVT.
    """
    print("\n4. SERIALIZACION DEL MODELO (.PTH) E INFERENCIA EN PRODUCCION")
    print("-" * 65)

    checkpoint = {
        "modelo_state_dict": modelo.state_dict(),
        "columnas": COLUMNAS_CARACTERISTICAS,
        "scaler_mean": escalador.mean_,
        "scaler_scale": escalador.scale_,
        "arquitectura": "Linear(5->64->32->16->1)_BatchNorm_Dropout"
    }
    torch.save(checkpoint, ARCHIVO_MODELO_PTH)
    print(f"Checkpoint PyTorch guardado en: {ARCHIVO_MODELO_PTH}")

    # Recargar modelo desde disco para simular inferencia en producción
    datos_cargados = torch.load(ARCHIVO_MODELO_PTH, map_location=DISPOSITIVO, weights_only=False)
    modelo_prod = RedNeuronalVeloxUTVT(num_entradas=len(datos_cargados["columnas"])).to(DISPOSITIVO)
    modelo_prod.load_state_dict(datos_cargados["modelo_state_dict"])
    modelo_prod.eval()

    escenarios = pd.DataFrame([
        {
            "ruta": "Terminal Toluca -> Campus UTVT (Tarifa accesible)",
            "distancia_km": 22.5, "costo": 15.0, "asientos_disponibles": 3,
            "puntualidad": 95.0, "calificacion_conductor": 4.9
        },
        {
            "ruta": "Lerma Centro -> Campus UTVT (Tarifa económica)",
            "distancia_km": 9.5, "costo": 12.0, "asientos_disponibles": 2,
            "puntualidad": 90.0, "calificacion_conductor": 4.7
        },
        {
            "ruta": "Metepec -> Campus UTVT (Tarifa elevada / 1 asiento)",
            "distancia_km": 19.0, "costo": 38.0, "asientos_disponibles": 1,
            "puntualidad": 68.0, "calificacion_conductor": 3.5
        }
    ])

    X_nuevos = escenarios[COLUMNAS_CARACTERISTICAS].to_numpy(dtype=float)
    X_nuevos_esc = (X_nuevos - datos_cargados["scaler_mean"]) / datos_cargados["scaler_scale"]

    with torch.no_grad():
        tensor_in = torch.tensor(X_nuevos_esc, dtype=torch.float32).to(DISPOSITIVO)
        probs = torch.sigmoid(modelo_prod(tensor_in)).cpu().numpy().flatten()

    print("\nResultados de Inferencia sobre Rutas Velox UTVT:")
    for idx, fila in escenarios.iterrows():
        prob_pct = probs[idx] * 100
        estado_pred = "VIABLE / CONSOLIDABLE" if probs[idx] >= 0.5 else "RIESGO OPERATIVO"
        print(f"  - {fila['ruta']}")
        print(f"    Costo: ${fila['costo']:.2f} | Asientos: {int(fila['asientos_disponibles'])} -> Probabilidad: {prob_pct:.2f}% ({estado_pred})")


# ============================================================
# 7. PUNTO DE ENTRADA
# ============================================================

def main():
    print("=" * 65)
    print("VELOX UTVT - PRACTICA 27 (ETAPA 19)")
    print("DEEP LEARNING EN PRODUCCION CON PYTORCH")
    print("=" * 65)

    df = cargar_dataset_profundo()

    X = df[COLUMNAS_CARACTERISTICAS].to_numpy(dtype=float)
    y = df["viaje_exitoso"].to_numpy(dtype=int)

    # Partición 70% entrenamiento, 15% validación, 15% prueba
    X_train_full, X_test, y_train_full, y_test = train_test_split(
        X, y, test_size=0.15, random_state=SEMILLA, stratify=y
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full, y_train_full, test_size=0.1765, random_state=SEMILLA, stratify=y_train_full
    )

    escalador = StandardScaler()
    X_train_esc = escalador.fit_transform(X_train)
    X_val_esc = escalador.transform(X_val)
    X_test_esc = escalador.transform(X_test)

    modelo, historial = entrenar_modelo_pytorch(X_train_esc, y_train, X_val_esc, y_val)

    # Evaluación final en conjunto de prueba (15%)
    print("\n3. EVALUACION EN CONJUNTO DE PRUEBA INDEPENDIENTE (15%)")
    print("-" * 65)

    modelo.eval()
    with torch.no_grad():
        tensor_test = torch.tensor(X_test_esc, dtype=torch.float32).to(DISPOSITIVO)
        probs_test = torch.sigmoid(modelo(tensor_test)).cpu().numpy().flatten()
        preds_test = (probs_test >= 0.5).astype(int)

    acc = accuracy_score(y_test, preds_test)
    prec = precision_score(y_test, preds_test, zero_division=0)
    rec = recall_score(y_test, preds_test, zero_division=0)
    f1 = f1_score(y_test, preds_test, zero_division=0)
    auc = roc_auc_score(y_test, probs_test)

    print(f"Accuracy:  {acc * 100:.2f}%")
    print(f"Precision: {prec * 100:.2f}%")
    print(f"Recall:    {rec * 100:.2f}%")
    print(f"F1-Score:  {f1 * 100:.2f}%")
    print(f"AUC-ROC:   {auc:.4f}")
    print("\nReporte de Clasificación:")
    print(classification_report(y_test, preds_test, target_names=["No Exitoso", "Exitoso"], zero_division=0))

    df_metricas = pd.DataFrame([{
        "modelo": "Red Neuronal Profunda PyTorch (64-32-16-1)",
        "accuracy": round(float(acc), 4),
        "precision": round(float(prec), 4),
        "recall": round(float(rec), 4),
        "f1_score": round(float(f1), 4),
        "auc_roc": round(float(auc), 4),
        "epocas": EPOCAS,
        "dispositivo": str(DISPOSITIVO)
    }])
    df_metricas.to_csv(REPORTE_METRICAS_CSV, index=False, encoding="utf-8-sig")

    generar_graficas_pytorch(historial, y_test, preds_test)
    guardar_y_probar_inferencia(modelo, escalador)

    print("\n5. ARCHIVOS GENERADOS EN 'resultados_dl_velox/'")
    print("-" * 65)
    print(f"- Modelo PyTorch (.pth): {ARCHIVO_MODELO_PTH}")
    print(f"- Métricas CSV:          {REPORTE_METRICAS_CSV}")
    print(f"- Curvas entrenamiento:  {GRAFICA_CURVAS}")
    print(f"- Matriz de confusión:   {GRAFICA_MATRIZ}")
    print("\nPRACTICA 27 FINALIZADA CORRECTAMENTE - SERIE VELOX UTVT COMPLETA.")


if __name__ == "__main__":
    main()