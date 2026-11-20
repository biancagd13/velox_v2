# ============================================================
# PRÁCTICA 3 ADAPTADA A VELOX UTVT: MODELO ML VS REGLAS FIJAS
# ============================================================

import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

def entrenar_modelo_velox():
    print("=" * 40)
    print("1. DATOS DE LOS USUARIOS / VIAJES EN VELOX")
    print("=" * 40)
    
    # Dataset extendido simulando la operación de Velox UTVT
    datos_entrenamiento = {
        "viajes_compartidos": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 2, 4, 6, 8, 10, 3, 5, 7, 9, 6],
        "puntualidad": [40, 50, 55, 60, 65, 70, 75, 80, 85, 90, 45, 55, 60, 65, 70, 75, 80, 85, 90, 95, 50, 60, 70, 80, 95, 55, 75, 85, 90, 68],
        "calificacion_conductor": [30, 40, 45, 55, 60, 65, 70, 80, 90, 95, 20, 35, 50, 60, 65, 70, 75, 85, 90, 100, 40, 55, 75, 85, 100, 45, 70, 80, 90, 60],
        "viaje_exitoso": [0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 0, 1, 1, 1, 0] # 1 = Exitoso, 0 = No viable
    }
    
    df = pd.DataFrame(datos_entrenamiento)
    
    print(df)
    
    print("\nCantidad de filas y columnas:")
    print(df.shape)
    
    print("\nValores faltantes:")
    print(df.isnull().sum())
    
    print("\nDistribucion de la variable objetivo (viaje_exitoso):")
    print(df["viaje_exitoso"].value_counts())
    
    # ------------------------------------------------------------
    # 1. DEFINIR VARIABLES DE ENTRADA (X) Y OBJETIVO (y)
    # ------------------------------------------------------------
    X = df[["viajes_compartidos", "puntualidad", "calificacion_conductor"]]
    y = df["viaje_exitoso"]

    print("\nCaracteristicas utilizadas (X):")
    print(X.head())

    print("\nResultados que queremos predecir (y):")
    print(y.head())

    print("\n" + "=" * 40)
    print("2. DIVISION DE LOS DATOS")
    print("=" * 40)

    # ------------------------------------------------------------
    # 2. DIVIDIR DATOS EN ENTRENAMIENTO Y PRUEBA 
    # ------------------------------------------------------------
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.30, random_state=42
    )

    print(f"Registros para entrenar: {len(X_train)}")
    print(f"Registros para probar: {len(X_test)}")

    print("\n" + "=" * 40)
    print("3. MODELO ENTRENADO")
    print("=" * 40)

    # ------------------------------------------------------------
    # 3. ENTRENAR MODELO DE MACHINE LEARNING (ÁRBOL DE DECISIONES)
    # ------------------------------------------------------------
    modelo = DecisionTreeClassifier(max_depth=3, random_state=42)
    modelo.fit(X_train, y_train)

    print("El arbol de decisiones termino su entrenamiento.")

    # ------------------------------------------------------------
    # 4. REGLA FIJA MANUAL PARA COMPARAR
    # ------------------------------------------------------------
    def predecir_con_reglas(fila):
        # Regla institucional de Velox: Puntualidad >= 70 y calificación >= 65
        if fila["puntualidad"] >= 70 and fila["calificacion_conductor"] >= 65:
            return 1
        return 0

    predicciones_reglas = X_test.apply(predecir_con_reglas, axis=1)
    predicciones_ml = modelo.predict(X_test)

    # Tabla comparativa en prueba
    df_comparacion = X_test.copy()
    df_comparacion["resultado_real"] = y_test
    df_comparacion["prediccion_reglas"] = predicciones_reglas
    df_comparacion["prediccion_ml"] = predicciones_ml

    print("\n" + "=" * 40)
    print("4. COMPARACION DE PREDICCIONES")
    print("=" * 40)
    print(df_comparacion.head(10))

    # ------------------------------------------------------------
    # 5. EVALUAR Y COMPARAR RESULTADOS
    # ------------------------------------------------------------
    exactitud_reglas = accuracy_score(y_test, predicciones_reglas)
    exactitud_ml = accuracy_score(y_test, predicciones_ml)

    print("\n" + "=" * 40)
    print("5. EXACTITUD DE LOS SISTEMAS")
    print("=" * 40)
    print(f"Exactitud de reglas fijas: {exactitud_reglas:.2%}")
    print(f"Exactitud de Machine Learning: {exactitud_ml:.2%}")

    print("\n" + "=" * 40)
    print("6. REPORTE DE CLASIFICACION")
    print("=" * 40)
    print(classification_report(y_test, predicciones_ml, target_names=["No exitoso", "Exitoso"]))

    print("=" * 40)
    print("7. MATRIZ DE CONFUSION")
    print("=" * 40)
    print(confusion_matrix(y_test, predicciones_ml))

    print("\n" + "=" * 40)
    print("8. PREDICCION PARA UN NUEVO USUARIO / TRAYECTO")
    print("=" * 40)
    # Nuevo caso de prueba en Velox
    nuevo_viaje = pd.DataFrame([[6, 80, 75]], columns=["viajes_compartidos", "puntualidad", "calificacion_conductor"])
    
    res_reglas = "Exitoso" if predecir_con_reglas(nuevo_viaje.iloc[0]) == 1 else "No exitoso"
    res_ml = "Exitoso" if modelo.predict(nuevo_viaje)[0] == 1 else "No exitoso"

    print(f"Viajes compartidos: {nuevo_viaje['viajes_compartidos'].values[0]}")
    print(f"Puntualidad: {nuevo_viaje['puntualidad'].values[0]}%")
    print(f"Calificación de conductor: {nuevo_viaje['calificacion_conductor'].values[0]}")
    print(f"Resultado con reglas: {res_reglas}")
    print(f"Resultado con Machine Learning: {res_ml}")

    print("\n" + "=" * 40)
    print("9. CONCLUSIONES")
    print("=" * 40)
    print("El sistema de reglas utiliza condiciones definidas manualmente.")
    print("El modelo de Machine Learning utiliza patrones aprendidos durante el entrenamiento.")
    print("Una mayor exactitud en esta prueba no garantiza que un sistema funcione mejor con datos futuros.")
    print("Para obtener conclusiones confiables necesitamos mas datos representativos y una evaluacion rigurosa.")
    
    print("\nPractica terminada correctamente.")

if __name__ == "__main__":
    entrenar_modelo_velox()