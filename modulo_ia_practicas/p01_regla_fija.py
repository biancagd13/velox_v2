# ============================================================
# PARTE B: TOMAR UNA DECISION MEDIANTE UNA REGLA FIJA
# ============================================================

# La siguiente informacion es ficticia.
# Las horas corresponden al tiempo de estudio semanal.

#Ahora resolveremos un problema diferente: 
#clasificar el resultado académico de un estudiante a partir de sus horas de estudio.

#Para comprender primero la diferencia conceptual, 
#utilizaremos una regla fija. Posteriormente, en la práctica de KNN,
#sustituiremos esa regla por un modelo que aprende de ejemplos.

viajes_velox = [
    {"id_viaje": 1, "costo": 18, "asientos_disponibles": 3},
    {"id_viaje": 2, "costo": 45, "asientos_disponibles": 1},
    {"id_viaje": 3, "costo": 15, "asientos_disponibles": 4},
    {"id_viaje": 4, "costo": 60, "asientos_disponibles": 0}
]


# ------------------------------------------------------------
# 1. DEFINIR UNA REGLA MANUAL O INSTITUCIONAL
# ------------------------------------------------------------
# Esta regla es solamente un ejemplo didáctico de lógica fija.
# No representa un modelo de optimización basado en entrenamiento.

def predecir_viabilidad_con_regla(costo):

    if costo <= 30:
        return "Viaje económico y viable"

    return "Viaje de alto costo / Revisar tarifa"


# ------------------------------------------------------------
# 2. APLICAR LA REGLA A CADA VIAJE REGISTRADO
# ------------------------------------------------------------
print("=== RESULTADO DE LA REGLA FIJA EN VELOX UTVT ===")

for viaje in viajes_velox:

    costo_viaje = viaje["costo"]
    resultado = predecir_viabilidad_con_regla(costo_viaje)

    print(
        f"Viaje ID #{viaje['id_viaje']}",
        f"| Costo: ${costo_viaje} MXN",
        f"| Asientos: {viaje['asientos_disponibles']}",
        f"| Estatus: {resultado}"
    )


# ------------------------------------------------------------
# 3. PROBAR CON UN CASO NUEVO EN LA PLATAFORMA
# ------------------------------------------------------------
costo_nuevo_viaje = 22

resultado_nuevo = predecir_viabilidad_con_regla(costo_nuevo_viaje)

print("\n=== NUEVO CASO EVALUADO EN VELOX ===")
print("Costo del trayecto:", f"${costo_nuevo_viaje} MXN")
print("Diagnóstico de regla fija:", resultado_nuevo)