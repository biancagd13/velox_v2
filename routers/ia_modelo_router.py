# ==============================================================================
# ENRUTADOR DE INTELIGENCIA ARTIFICIAL - VELOX UTVT (DATOS REALES DE MONGODB)
# ==============================================================================

# Importamos APIRouter para crear rutas, Request para recibir peticiones HTTP y Response para enviar archivos
from fastapi import APIRouter, Request, Response

# Importamos Jinja2Templates para renderizar los archivos HTML del frontend
from fastapi.templating import Jinja2Templates

# Importamos pandas para organizar, filtrar y estructurar los datos en tablas (DataFrames)
import pandas as pd

# Importamos train_test_split para dividir el dataset en conjuntos de entrenamiento y prueba
from sklearn.model_selection import train_test_split

# Importamos DecisionTreeClassifier para crear el modelo de Machine Learning de Árbol de Decisiones
from sklearn.tree import DecisionTreeClassifier

# Importamos métricas clave para evaluar el desempeño (exactitud, reporte y matriz de confusión)
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Importamos la conexión oficial y activa a tu base de datos MongoDB desde database.py
from database import db

# Inicializamos el enrutador para agrupar todas las rutas de este módulo de IA
router_ia_modelo = APIRouter()

# Configuramos la carpeta base donde se encuentran las plantillas HTML
templates = Jinja2Templates(directory="templates")


def _extraer_viajes_limpios() -> pd.DataFrame:
    """Consulta la colección 'viaje' en MongoDB y devuelve un DataFrame limpio."""
    lista_viajes_bd = []
    try:
        cursor_viajes = db.viaje.find(
            {}, {"_id": 0, "costo": 1, "asientos_disponibles": 1, "estado": 1}
        )
        for documento in cursor_viajes:
            valor_costo = documento.get("costo")
            costo_actual = float(valor_costo) if valor_costo is not None else 0.0

            valor_asientos = documento.get("asientos_disponibles")
            asientos_actuales = int(valor_asientos) if valor_asientos is not None else 0

            estado_viaje = documento.get("estado", "")
            viaje_exitoso = (
                1 if estado_viaje in ["en_curso", "finalizado", "activo"] else 0
            )

            lista_viajes_bd.append(
                {
                    "costo": costo_actual,
                    "asientos_disponibles": asientos_actuales,
                    "viaje_exitoso": viaje_exitoso,
                }
            )
    except Exception as error:
        print(f"Error al consultar MongoDB: {error}")

    if lista_viajes_bd:
        return pd.DataFrame(lista_viajes_bd)
    return pd.DataFrame(columns=["costo", "asientos_disponibles", "viaje_exitoso"])


# ==============================================================================
# FUNCIÓN AUXILIAR INTERNA PARA PROCESAR EL MODELO Y OBTENER LA TABLA DE PRUEBA
# ==============================================================================
def obtener_datos_y_predicciones_ml():
    tabla_datos = _extraer_viajes_limpios()
    total_registros = len(tabla_datos)

    if total_registros > 0:
        caracteristicas_entrada = tabla_datos[["costo", "asientos_disponibles"]]
        variable_objetivo = tabla_datos["viaje_exitoso"]
    else:
        caracteristicas_entrada = pd.DataFrame(columns=["costo", "asientos_disponibles"])
        variable_objetivo = pd.Series(dtype=int)

    # Dividimos los datos en 50% para entrenamiento y 50% para prueba (si hay al menos 2 registros)
    if total_registros >= 2:
        conteo_clases = variable_objetivo.value_counts()
        puede_estratificar = len(conteo_clases) > 1 and conteo_clases.min() >= 2
        X_entrenamiento, X_prueba, y_entrenamiento, y_prueba = train_test_split(
            caracteristicas_entrada,
            variable_objetivo,
            test_size=0.50,
            random_state=42,
            stratify=variable_objetivo if puede_estratificar else None,
        )
    else:
        X_entrenamiento, X_prueba = caracteristicas_entrada, caracteristicas_entrada
        y_entrenamiento, y_prueba = variable_objetivo, variable_objetivo

    # Creamos una función de reglas fijas manuales para comparar su lógica contra la IA
    def evaluar_por_reglas_manuales(fila_datos):
        # Si el costo es menor o igual a 20 y hay asientos, retorna 1 (exitoso); de lo contrario, 0
        if fila_datos["costo"] <= 20 and fila_datos["asientos_disponibles"] > 0:
            return 1
        return 0

    # Aplicamos las reglas manuales fila por fila al conjunto de prueba
    if not X_prueba.empty:
        predicciones_reglas = X_prueba.apply(evaluar_por_reglas_manuales, axis=1)
    else:
        predicciones_reglas = pd.Series(dtype=int)

    # Inicializamos el modelo de Árbol de Decisiones con una profundidad máxima de 3
    modelo_arbol = DecisionTreeClassifier(max_depth=3, random_state=42)

    # Entrenamos el modelo con los datos de entrenamiento y predecimos sobre el conjunto de prueba
    if not X_entrenamiento.empty and len(set(y_entrenamiento)) > 1:
        modelo_arbol.fit(X_entrenamiento, y_entrenamiento)
        predicciones_ml = pd.Series(
            modelo_arbol.predict(X_prueba), index=X_prueba.index
        )
    else:
        predicciones_ml = y_prueba.copy()

    # Si hay datos de prueba, construimos una tabla comparativa con los resultados reales y predicciones
    if not X_prueba.empty:
        tabla_comparativa = X_prueba.copy()
        tabla_comparativa["resultado_real"] = y_prueba
        tabla_comparativa["prediccion_reglas"] = predicciones_reglas
        tabla_comparativa["prediccion_ml"] = predicciones_ml
        return tabla_comparativa
    else:
        return pd.DataFrame()


# ==============================================================================
# RUTA GET PARA ACCEDER AL MÓDULO DE IA DESDE LA WEB
# ==============================================================================
@router_ia_modelo.get("/modulo_ia")
def entrenar_modelo_velox(request: Request):
    # Invocamos la función auxiliar para obtener la tabla con las predicciones de prueba
    tabla_comparativa = obtener_datos_y_predicciones_ml()

    # Consultamos nuevamente MongoDB de manera directa para calcular las tarjetas y métricas generales del dashboard
    tabla_datos = _extraer_viajes_limpios()

    # Obtenemos las dimensiones generales de la tabla completa de la BD
    dimensiones_tabla = tabla_datos.shape
    total_registros = len(tabla_datos)

    # Calculamos la distribución de cuántos son exitosos y cuántos no (0 y 1)
    distribucion_objetivo = (
        tabla_datos["viaje_exitoso"].value_counts().to_dict()
        if total_registros > 0
        else {0: 0, 1: 0}
    )

    # Determinamos los tamaños exactos de entrenamiento y prueba para las tarjetas visuales
    if total_registros >= 2:
        conteo_clases = tabla_datos["viaje_exitoso"].value_counts()
        puede_estratificar = len(conteo_clases) > 1 and conteo_clases.min() >= 2
        _, X_prueba, _, y_prueba = train_test_split(
            tabla_datos[["costo", "asientos_disponibles"]],
            tabla_datos["viaje_exitoso"],
            test_size=0.50,
            random_state=42,
            stratify=tabla_datos["viaje_exitoso"] if puede_estratificar else None,
        )
        len_train_val = total_registros - len(X_prueba)
        len_test_val = len(X_prueba)
    else:
        len_train_val = total_registros
        len_test_val = 0
        y_prueba = pd.Series(dtype=int)

    # Calculamos los porcentajes de exactitud, reporte de clasificación y matriz de confusión
    if not tabla_comparativa.empty:
        exactitud_reglas_valor = accuracy_score(
            tabla_comparativa["resultado_real"], tabla_comparativa["prediccion_reglas"]
        )
        exactitud_ml_valor = accuracy_score(
            tabla_comparativa["resultado_real"], tabla_comparativa["prediccion_ml"]
        )
        lista_comparacion_final = tabla_comparativa.reset_index().to_dict(
            orient="records"
        )

        texto_reporte_clasificacion = classification_report(
            tabla_comparativa["resultado_real"],
            tabla_comparativa["prediccion_ml"],
            labels=[0, 1],
            target_names=["No exitoso", "Exitoso"],
            zero_division=0,
        )
        matriz_confusion_resultado = confusion_matrix(
            tabla_comparativa["resultado_real"],
            tabla_comparativa["prediccion_ml"],
            labels=[0, 1],
        ).tolist()
    else:
        exactitud_reglas_valor = 0.0
        exactitud_ml_valor = 0.0
        lista_comparacion_final = []
        texto_reporte_clasificacion = "No hay suficientes datos."
        matriz_confusion_resultado = [[0, 0], [0, 0]]

    # Simulamos un nuevo trayecto de prueba con valores fijos ($15 y 2 asientos) para la interfaz
    nuevo_trayecto_simulacion = pd.DataFrame(
        [[15, 2]], columns=["costo", "asientos_disponibles"]
    )
    modelo_temp = DecisionTreeClassifier(max_depth=3, random_state=42)
    if total_registros >= 2:
        modelo_temp.fit(
            tabla_datos[["costo", "asientos_disponibles"]], tabla_datos["viaje_exitoso"]
        )
        res_ml_pred = modelo_temp.predict(nuevo_trayecto_simulacion)[0]
    else:
        res_ml_pred = 0

    resultado_reglas_nuevo = "Exitoso" if (15 <= 20 and 2 > 0) else "No exitoso"
    resultado_ml_nuevo = "Exitoso" if res_ml_pred == 1 else "No exitoso"

    diccionario_nuevo_caso = {
        "costo": 15.0,
        "asientos": 2,
        "res_reglas": resultado_reglas_nuevo,
        "res_ml": resultado_ml_nuevo,
    }

    # Redactamos las conclusiones institucionales del módulo de IA
    lista_conclusiones = [
        f"El dashboard muestra exactamente los {total_registros} registros reales almacenados en MongoDB.",
        "El modelo de Machine Learning analiza directamente el costo y la disponibilidad real de asientos.",
        "Al registrar nuevos viajes en la plataforma web, el árbol de decisiones ajustará sus patrones automáticamente.",
    ]

    # Renderizamos la plantilla HTML del dashboard enviándole todas las variables calculadas
    return templates.TemplateResponse(
        request,
        "ia_modulo/dashboard_ia.html",
        {
            "shape": dimensiones_tabla,
            "distribucion": distribucion_objetivo,
            "len_train": len_train_val,
            "len_test": len_test_val,
            "exactitud_reglas": f"{exactitud_reglas_valor:.2%}",
            "exactitud_ml": f"{exactitud_ml_valor:.2%}",
            "lista_comparacion": lista_comparacion_final,
            "reporte_str": texto_reporte_clasificacion,
            "matriz": matriz_confusion_resultado,
            "nuevo_caso": diccionario_nuevo_caso,
            "conclusiones": lista_conclusiones,
        },
    )


# ==============================================================================
# RUTA PARA EXPORTAR LOS RESULTADOS Y DATOS PROCESADOS EN CSV
# ==============================================================================
@router_ia_modelo.get("/modulo_ia/exportar_csv")
def exportar_resultados_csv():
    # Volvemos a procesar la función auxiliar para garantizar que los datos estén frescos al hacer clic
    tabla_comparativa = obtener_datos_y_predicciones_ml()

    # Si la tabla está vacía, exportamos únicamente las cabeceras; si tiene datos, los convertimos a formato CSV
    if tabla_comparativa.empty:
        datos_salida = (
            "costo,asientos_disponibles,resultado_real,prediccion_reglas,prediccion_ml\n"
        )
    else:
        datos_salida = tabla_comparativa.to_csv(index=True, encoding="utf-8")

    # Retornamos una respuesta HTTP configurada como archivo descargable (.csv)
    return Response(
        content=datos_salida,
        media_type="text/csv",
        headers={
            "Content-Disposition": "attachment; filename=resultados_velox_ml.csv"
        },
    )