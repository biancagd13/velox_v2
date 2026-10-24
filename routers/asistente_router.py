# ==========================================================
# MÓDULO DEL ASISTENTE VIRTUAL IA (LOGISMART)
# Proyecto: Velox UTVT
# ==========================================================

from datetime import datetime

# Importamos la librería ollama para la comunicación local con los modelos de lenguaje (Llama 3.2)
import ollama

# Importamos la conexión centralizada a la base de datos de MongoDB
from database import db

# Importamos herramientas de FastAPI para enrutamiento, peticiones HTTP y respuestas
from fastapi import APIRouter, Request, status
from fastapi.responses import JSONResponse
from fastapi.templating import Jinja2Templates

# Inicializamos el enrutador para el módulo del asistente con su prefijo institucional y etiqueta
router_asistente = APIRouter(prefix="/asistente", tags=["Asistente IA"])

# Configuramos el motor de plantillas HTML apuntando a la carpeta 'templates' del proyecto
motor_plantillas = Jinja2Templates(directory="templates")

# Definimos el nombre del modelo de lenguaje local que utilizará Ollama
nombre_modelo_llm = "llama3.2"

# Almacenamiento temporal en memoria para el historial de mensajes de la sesión del chat
historial_conversacion_chat = []


def obtener_contexto_operativo_velox():
    """Consulta en tiempo real MongoDB para extraer el estado operativo, métricas de confianza, ecológicas y de puntualidad."""
    try:
        # Contamos usuarios y vehículos registrados
        total_usuarios_registrados = db["usuario"].count_documents({})
        total_vehiculos_registrados = db["vehiculo"].count_documents({})
        
        # Obtenemos las rutas y viajes activos
        lista_rutas_activas = list(db["ruta"].find({}, {"_id": 0}))
        lista_viajes_disponibles = list(db["viaje"].find({"asientos_disponibles": {"$gt": 0}}, {"_id": 0}))
        lista_todos_los_viajes = list(db["viaje"].find({}, {"_id": 0}))
        
        # Consultamos calificaciones generales
        lista_calificaciones = list(db["calificacion"].find({}, {"_id": 0, "puntuacion": 1, "id_conductor": 1}))
        promedio_calificacion_servicio = (
            sum([calificacion.get("puntuacion", 0) for calificacion in lista_calificaciones]) / len(lista_calificaciones)
            if lista_calificaciones else 5.0
        )

        # ----------------------------------------------------------
        # 1. CÁLCULO DE AHORRO ECONÓMICO Y CO₂
        # ----------------------------------------------------------
        # Tarifa media en taxi/Uber estimado hacia UTVT = $110 MXN. En Velox promedio = $18 MXN.
        costo_promedio_carpooling = (
            sum([v.get("costo", 18) for v in lista_viajes_disponibles]) / len(lista_viajes_disponibles)
            if lista_viajes_disponibles else 18.0
        )
        ahorro_estimado_viaje = max(0.0, 110.0 - costo_promedio_carpooling)
        co2_evitado_por_viaje_kg = 2.1  # Promedio de kg de CO2 evitados por compartir auto en viajes interurbanos Toluca-UTVT

        # ----------------------------------------------------------
        # 2. ANÁLISIS DE DEMANDA Y CONFIANZA POR CONDUCTOR/RUTA
        # ----------------------------------------------------------
        demanda_por_ruta = {}
        for viaje in lista_todos_los_viajes:
            id_r = viaje.get("id_ruta")
            demanda_por_ruta[id_r] = demanda_por_ruta.get(id_r, 0) + 1

        # Construimos el texto resumen operativo con datos ampliados
        resumen_contexto = f"--- ESTADO ACTUAL Y MÉTRICAS EN TIEMPO REAL VELOX UTVT ---\n"
        resumen_contexto += f"- Usuarios registrados: {total_usuarios_registrados}\n"
        resumen_contexto += f"- Vehículos activos en sistema: {total_vehiculos_registrados}\n"
        resumen_contexto += f"- Calificación promedio general: {promedio_calificacion_servicio:.1f} estrellas\n"
        resumen_contexto += f"- Ahorro promedio estimado por usuario frente a taxi/Uber: ${ahorro_estimado_viaje:.1f} MXN por trayecto\n"
        resumen_contexto += f"- Reducción de huella de carbono estimada: {co2_evitado_por_viaje_kg} kg de CO₂ por cada asiento ocupado\n\n"
        
        resumen_contexto += "RUTAS Y HORARIOS DISPONIBLES:\n"
        for ruta in lista_rutas_activas:
            id_r = ruta.get('id_ruta')
            conteo_v = demanda_por_ruta.get(id_r, 0)
            resumen_contexto += (
                f"ID Ruta #{id_r}: {ruta.get('origen')} hacia {ruta.get('destino')} "
                f"(Hora oficial de salida: {ruta.get('hora_salida', '07:00 AM')}) | Total viajes programados: {conteo_v}\n"
            )
            
        resumen_contexto += "\nVIAJES PROGRAMADOS CON ASIENTOS LIBRES:\n"
        for viaje in lista_viajes_disponibles:
            resumen_contexto += (
                f"Viaje ID #{viaje.get('id_viaje')} | Ruta #{viaje.get('id_ruta')} | "
                f"Fecha: {viaje.get('fecha')} | Asientos libres: {viaje.get('asientos_disponibles')} | "
                f"Costo: ${viaje.get('costo')} MXN | Estado: {viaje.get('estado')}\n"
            )

        return resumen_contexto
    except Exception as error_conexion:
        return f"[Aviso: No se pudo cargar el contexto en tiempo real: {error_conexion}]"


def generar_instruccion_sistema():
    """Genera las instrucciones de comportamiento, personalidad, tono mexicano y reglas para las nuevas funciones de IA."""
    datos_actuales_sistema = obtener_contexto_operativo_velox()
    return f"""
Eres LogiSmart, un Asistente Experto y Coordinador Logístico Inteligente de la plataforma de carpooling Velox UTVT.

Tu función es orientar al usuario en:
1. Recomendación de rutas y viajes activos según los asientos disponibles, costos y horarios.
2. Análisis de calificaciones, Nivel de Confianza y Verificación de Conductores.
3. Calculadora de Ahorro Económico ($ MXN) y Emisión de Carbono (kg de CO₂ evitados).
4. Alertas de Puntualidad y Hora Límite de Salida (sugiriendo salir con 15-20 min de margen previo a la clase en UTVT).
5. Generación de diagnósticos de demanda y rutas con alta frecuencia.

REGLAS ESTRICTAS:
- Basa tus respuestas en los datos reales del sistema proporcionados a continuación.
- Queda PROHIBIDO hablar de temas ajenos a movilidad, transporte, carpooling o la UTVT.
- Dirígete siempre a la persona como 'usuario' o por su nombre si te lo da. NUNCA uses la palabra 'estudiante'.
- NO utilices emojis bajo ninguna circunstancia.
- Sé alegre, dinámica y utiliza frases divertidas mexicanas combinadas de forma natural (Ejemplos: ¡Me late chocolate!, ¡Recórcholis!, ¡Reconchinchiles!, ¡Me late cacahuate!, ¡Me canso ganso!, ¡A todo dar!, ¡Súper chido!, ¡De a devis!).
- SI EL USUARIO PREGUNTA QUÉ PUEDES HACER O TUS FUNCIONES: Responde siempre presentando una lista ordenada de viñetas claras como esta:
  "¡A todo dar! Como LogiSmart, te puedo ayudar con todo esto:
   • Recomendación de viajes económicos y rutas activas en tiempo real.
   • Verificación de Nivel de Confianza de conductores y vehículos.
   • Calculadora de ahorro en dinero ($ MXN) y reducción de huella de carbono (kg de CO₂).
   • Asistente de Puntualidad y cálculo de hora límite de salida hacia la UTVT.
   • Generación de diagnósticos de demanda de rutas para coordinadores."
- Responde de forma compacta y clara. Evita saltos de línea innecesarios o renglones vacíos dobles.

{datos_actuales_sistema}
"""


@router_asistente.get("/")
def vista_asistente(peticion_http: Request):
    """Ruta HTTP GET para renderizar la vista principal del módulo de asistente IA dentro de la interfaz web."""
    global historial_conversacion_chat
    if not historial_conversacion_chat:
        historial_conversacion_chat = [{"role": "system", "content": generar_instruccion_sistema()}]
    
    return motor_plantillas.TemplateResponse(
        peticion_http,
        "asistente/asistente_inicio.html",
        {}
    )


@router_asistente.post("/consultar")
async def consultar_asistente(peticion_http: Request):
    """Ruta HTTP POST para procesar las consultas asíncronas del usuario y retornar la respuesta generada por Ollama."""
    try:
        cuerpo_peticion = await peticion_http.json()
        pregunta_usuario = cuerpo_peticion.get("mensaje", "").strip()
        
        if not pregunta_usuario:
            return JSONResponse({"error": "Mensaje vacío"}, status_code=400)

        # Actualizamos el contexto de MongoDB en tiempo real
        if historial_conversacion_chat and historial_conversacion_chat[0]["role"] == "system":
            historial_conversacion_chat[0]["content"] = generar_instruccion_sistema()
        else:
            historial_conversacion_chat.insert(0, {"role": "system", "content": generar_instruccion_sistema()})

        historial_conversacion_chat.append({"role": "user", "content": pregunta_usuario})

        respuesta_modelo_ollama = ollama.chat(
            model=nombre_modelo_llm,
            messages=historial_conversacion_chat
        )

        contenido_respuesta_ia = respuesta_modelo_ollama["message"]["content"]
        historial_conversacion_chat.append({"role": "assistant", "content": contenido_respuesta_ia})

        return JSONResponse({"respuesta": contenido_respuesta_ia})

    except Exception as error_procesamiento:
        if historial_conversacion_chat and historial_conversacion_chat[-1]["role"] == "user":
            historial_conversacion_chat.pop()
        return JSONResponse({"error": str(error_procesamiento)}, status_code=500)


@router_asistente.post("/reiniciar")
def reiniciar_asistente():
    """Reinicia la memoria temporal de la conversación del chat."""
    global historial_conversacion_chat
    historial_conversacion_chat = [{"role": "system", "content": generar_instruccion_sistema()}]
    return JSONResponse({"mensaje": "¡A todo dar! Memoria del asistente reiniciada correctamente."})


@router_asistente.get("/reporte-logistico")
def generar_reporte_logistico_administracion():
    """
    ENDPOINT ADMINISTRATIVO EXPORTABLE / REPORTE EN FORMATO LISTA:
    Genera un diagnóstico en tiempo real de las rutas con mayor demanda del día y las rutas
    que requieren mayor asignación de conductores para balancear el servicio.
    """
    try:
        lista_rutas = list(db["ruta"].find({}, {"_id": 0}))
        lista_viajes = list(db["viaje"].find({}, {"_id": 0}))

        conteo_viajes_por_ruta = {}
        asientos_totales_por_ruta = {}

        for viaje in lista_viajes:
            id_r = viaje.get("id_ruta")
            conteo_viajes_por_ruta[id_r] = conteo_viajes_por_ruta.get(id_r, 0) + 1
            asientos_totales_por_ruta[id_r] = asientos_totales_por_ruta.get(id_r, 0) + viaje.get("asientos_disponibles", 0)

        reporte_rutas = []
        for ruta in lista_rutas:
            id_r = ruta.get("id_ruta")
            origen = ruta.get("origen")
            destino = ruta.get("destino")
            total_viajes = conteo_viajes_por_ruta.get(id_r, 0)
            asientos_libres = asientos_totales_por_ruta.get(id_r, 0)

            # Lógica de diagnóstico para administradores
            if total_viajes == 0:
                estado_atencion = "URGENTE: Sin conductores asignados"
            elif asientos_libres <= 2:
                estado_atencion = "ALTA DEMANDA: Se recomiendan más vehículos"
            else:
                estado_atencion = "OPTIMO: Cobertura adecuada"

            reporte_rutas.append({
                "id_ruta": id_r,
                "trayecto": f"{origen} a {destino}",
                "viajes_programados": total_viajes,
                "asientos_disponibles": asientos_libres,
                "diagnostico_administracion": estado_atencion
            })

        # Ordenamos las rutas por mayor demanda de viajes
        reporte_rutas = sorted(reporte_rutas, key=lambda x: x["viajes_programados"], reverse=True)

        return JSONResponse({
            "titulo_reporte": "Reporte Oficial de Logística y Demanda Velox UTVT",
            "fecha_generacion": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "total_rutas_evaluadas": len(reporte_rutas),
            "diagnostico_rutas": reporte_rutas
        })
    except Exception as error_reporte:
        return JSONResponse({"error": f"Error al generar el reporte logístico: {error_reporte}"}, status_code=500)