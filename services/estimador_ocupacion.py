# ============================================================
# VELOX UTVT - ESTIMADOR DE OCUPACION Y LLENADO DE VIAJES
# Inspirado en p26 / p27 y conectado a MongoDB (db.viaje)
# ============================================================

import os
import sys
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

RAIZ_PROYECTO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if RAIZ_PROYECTO not in sys.path:
    sys.path.append(RAIZ_PROYECTO)

try:
    from database import db
except ImportError:
    db = None

SEMILLA = 42


def _extraer_viajes_ocupacion() -> pd.DataFrame:
    """
    Lee viajes desde db.viaje y calcula una estimacion de asientos ocupados.
    Si el documento trae 'capacidad_total' o 'asientos_totales', usa la diferencia;
    si no, estima capacidad base de 4 o 5 asientos segun asientos_disponibles.
    """
    filas: list[dict[str, float]] = []

    if db is not None:
        try:
            cursor = db.viaje.find(
                {},
                {
                    "_id": 0,
                    "costo": 1,
                    "asientos_disponibles": 1,
                    "capacidad_total": 1,
                    "asientos_totales": 1,
                    "estado": 1,
                },
            )
            for doc in cursor:
                costo = float(doc.get("costo") or 0.0)
                disp = int(float(doc.get("asientos_disponibles") or 0))
                cap_raw = doc.get("capacidad_total") or doc.get("asientos_totales")

                if cap_raw is not None:
                    capacidad = max(1, int(float(cap_raw)))
                else:
                    capacidad = max(4, disp + 1)

                ocupados = max(0, min(capacidad, capacidad - disp))
                if str(doc.get("estado") or "").lower() == "finalizado" and ocupados == 0:
                    ocupados = max(1, capacidad - disp)

                filas.append({
                    "costo": max(0.0, min(500.0, costo)),
                    "asientos_disponibles": max(0, min(6, disp)),
                    "capacidad_total": float(capacidad),
                    "asientos_ocupados": float(ocupados),
                })
        except Exception as error:
            print(f"[Aviso Estimador] Error leyendo db.viaje: {error}")

    df = pd.DataFrame(filas)
    if len(df) < 15:
        rng = np.random.default_rng(SEMILLA)
        faltantes = 120 - len(df)
        costos = rng.uniform(15.0, 120.0, faltantes)
        capacidades = rng.integers(3, 6, faltantes)
        # Costos moderados suelen tener mayor ocupacion
        factor_costo = np.clip(1.0 - (costos / 160.0), 0.15, 0.95)
        ocupados_sim = np.clip(
            np.round(capacidades * factor_costo + rng.normal(0, 0.5, faltantes)),
            0,
            capacidades,
        )
        disp_sim = np.maximum(0, capacidades - ocupados_sim)

        df_respaldo = pd.DataFrame({
            "costo": np.round(costos, 2),
            "asientos_disponibles": disp_sim.astype(float),
            "capacidad_total": capacidades.astype(float),
            "asientos_ocupados": ocupados_sim.astype(float),
        })
        df = pd.concat([df, df_respaldo], ignore_index=True)

    return df


def estimar_ocupacion_viaje(
    costo: float,
    asientos_disponibles: int,
    capacidad_total: int | None = None,
) -> dict[str, Any]:
    """
    Entrena un regresor sobre los datos disponibles y estima:
    - asientos_ocupados_estimados
    - porcentaje_llenado_estimado
    - nivel_demanda ('Alta', 'Media' o 'Baja')
    """
    df = _extraer_viajes_ocupacion()
    X = df[["costo", "asientos_disponibles", "capacidad_total"]]
    y = df["asientos_ocupados"]

    modelo = RandomForestRegressor(n_estimators=150, random_state=SEMILLA, n_jobs=-1)
    modelo.fit(X, y)

    disp_limpio = max(0, min(6, int(asientos_disponibles)))
    cap_limpia = max(1, int(capacidad_total)) if capacidad_total else max(4, disp_limpio + 1)
    costo_limpio = max(0.0, min(500.0, float(costo)))

    entrada = pd.DataFrame([{
        "costo": costo_limpio,
        "asientos_disponibles": float(disp_limpio),
        "capacidad_total": float(cap_limpia),
    }])

    pred_ocupados = float(modelo.predict(entrada)[0])
    pred_ocupados = round(max(0.0, min(float(cap_limpia), pred_ocupados)), 2)
    pct_llenado = round((pred_ocupados / cap_limpia) * 100.0, 1)

    if pct_llenado >= 70.0:
        nivel = "Alta"
    elif pct_llenado >= 40.0:
        nivel = "Media"
    else:
        nivel = "Baja"

    return {
        "costo": round(costo_limpio, 2),
        "asientos_disponibles": disp_limpio,
        "capacidad_total": cap_limpia,
        "asientos_ocupados_estimados": pred_ocupados,
        "porcentaje_llenado_estimado": pct_llenado,
        "nivel_demanda": nivel,
    }


if __name__ == "__main__":
    print(estimar_ocupacion_viaje(costo=25.0, asientos_disponibles=2, capacidad_total=4))