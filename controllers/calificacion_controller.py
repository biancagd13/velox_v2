from datetime import datetime
from database import db
from models.calificacion_model import CalificacionModel, CalificacionUpdateModel

# Referenciamos la colección específica en la base de datos
coleccion_calificacion = db["calificacion"]


class CalificacionController:
    """
    Controlador encargado de gestionar toda la lógica de negocio y operaciones CRUD
    para la entidad, comunicándose directamente con MongoDB de forma segura.
    """

    @staticmethod
    def obtener_siguiente_id():
        """Calcula y retorna el siguiente ID secuencial numérico para el registro."""
        try:
            # Buscamos el último documento registrado ordenado de forma descendente por el ID
            ultimo_documento = coleccion_calificacion.find_one(
                {"id_calificacion": {"$exists": True}}, sort=[("id_calificacion", -1)]
            )

            # Si existe un documento previo, incrementamos su ID en uno; de lo contrario, iniciamos en 1
            if ultimo_documento and "id_calificacion" in ultimo_documento:
                return ultimo_documento["id_calificacion"] + 1
            return 1

        except Exception as error:
            print(f"[ERROR EN TERMINAL] No se pudo calcular el siguiente ID de calificacion: {error}")
            return None

    @staticmethod
    def crear_calificacion(calificacion: CalificacionModel):
        """Inserta un nuevo registro en la base de datos tras validar sus datos."""
        try:
            # Obtenemos el ID correlativo único para el nuevo registro
            nuevo_id = CalificacionController.obtener_siguiente_id()
            if nuevo_id is None:
                print("[ALERTA] Operación cancelada: No se pudo generar el ID para el calificacion.")
                return None

            # Convertimos el modelo Pydantic a un diccionario plano para MongoDB
            documento_calificacion = calificacion.model_dump()
            documento_calificacion["id_calificacion"] = nuevo_id

            # Si no se proporciona una fecha de registro, asignamos la fecha y hora actual
            if not documento_calificacion.get("fecha_registro"):
                documento_calificacion["fecha_registro"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            # Insertamos el documento estructurado en la colección correspondiente
            coleccion_calificacion.insert_one(documento_calificacion)
            print(f"[ÉXITO] calificacion creada correctamente con ID: {nuevo_id}")
            
            return nuevo_id

        except Exception as error:
            print(f"[ERROR EN TERMINAL] Falló la creación del calificacion en MongoDB: {error}")
            return None

    @staticmethod
    def obtener_todas_las_calificaciones():
        """Recupera la lista completa de todas las calificaciones enriquecidas con el nombre del usuario y la ruta del viaje."""
        try:
            # Consultamos las calificaciones ordenadas por ID
            calificaciones = list(
                coleccion_calificacion.find({"id_calificacion": {"$exists": True}}, {"_id": 0}).sort("id_calificacion", 1)
            )

            # Recorremos cada calificación para buscar el usuario y el viaje correspondientes
            for calificacion in calificaciones:
                # 1. Resolver Nombre del Usuario
                id_usu = calificacion.get("id_usuario")
                if id_usu is not None:
                    usuario_doc = db["usuario"].find_one({"id_usuario": id_usu}, {"_id": 0, "nombre": 1, "apellido_paterno": 1})
                    if usuario_doc:
                        calificacion["nombre_usuario"] = f"{usuario_doc.get('nombre', '')} {usuario_doc.get('apellido_paterno', '')}".strip()
                    else:
                        calificacion["nombre_usuario"] = "Usuario Desconocido"
                else:
                    calificacion["nombre_usuario"] = "Sin Asignar"

                # 2. Resolver ID y Ruta del Viaje de forma robusta
                id_vj = calificacion.get("id_viaje")
                if id_vj is not None:
                    viaje_doc = db["viaje"].find_one({"id_viaje": id_vj}, {"_id": 0})
                    if viaje_doc:
                        # Verificamos si el viaje tiene origen/destino directo o nombres alternativos
                        origen = viaje_doc.get("origen") or viaje_doc.get("punto_partida")
                        destino = viaje_doc.get("destino") or viaje_doc.get("punto_llegada")

                        # Si el viaje tiene una relación con una colección de rutas (id_ruta) y no tiene origen directo
                        if not origen and "id_ruta" in viaje_doc:
                            ruta_doc = db["ruta"].find_one({"id_ruta": viaje_doc["id_ruta"]}, {"_id": 0})
                            if ruta_doc:
                                origen = ruta_doc.get("origen") or ruta_doc.get("punto_partida", "Origen")
                                destino = ruta_doc.get("destino") or ruta_doc.get("punto_llegada", "Destino")

                        # Construimos la descripción final si encontramos los datos
                        if origen and destino:
                            calificacion["descripcion_viaje"] = f"Viaje #{id_vj}: {origen} ➔ {destino}"
                        else:
                            calificacion["descripcion_viaje"] = f"Viaje #{id_vj}"
                    else:
                        calificacion["descripcion_viaje"] = f"Viaje #{id_vj}"
                else:
                    calificacion["descripcion_viaje"] = "Sin Viaje Asignado"

            return calificaciones

        except Exception as error:
            print(f"[ERROR EN TERMINAL] No se pudieron obtener las calificaciones enriquecidas: {error}")
            return []

    @staticmethod
    def obtener_calificacion_por_id(id_calificacion: int):
        """Busca y retorna un dato específico filtrando por su identificador único."""
        try:
            # Consultamos el registro que coincida con el ID indicado
            calificacion_encontrado = coleccion_calificacion.find_one({"id_calificacion": id_calificacion}, {"_id": 0})
            return calificacion_encontrado

        except Exception as error:
            print(f"[ERROR EN TERMINAL] Error al buscar la calificacion con ID {id_calificacion}: {error}")
            return None

    @staticmethod
    def actualizar_calificacion(id_calificacion: int, calificacion_update: CalificacionUpdateModel):
        """Actualiza de forma parcial los datos existentes según los campos enviados."""
        try:
            # Filtramos únicamente los campos que no sean nulos para evitar sobreescribir información
            datos_a_actualizar = {
                clave: valor for clave, valor in calificacion_update.model_dump().items() if valor is not None
            }

            if not datos_a_actualizar:
                print(f"[ALERTA] No se proporcionaron datos válidos para actualizar al calificacion {id_calificacion}.")
                return False

            # Ejecutamos la actualización en la base de datos utilizando el operador $set
            resultado = coleccion_calificacion.update_one(
                {"id_calificacion": id_calificacion}, 
                {"$set": datos_a_actualizar}
            )

            if resultado.matched_count > 0:
                print(f"[ÉXITO] calificacion con ID {id_calificacion} actualizado correctamente.")
                return True
            else:
                print(f"[ALERTA] No se encontró ningún calificacion con el ID {id_calificacion} para actualizar.")
                return False

        except Exception as error:
            print(f"[ERROR EN TERMINAL] Falló la actualización del calificacion {id_calificacion}: {error}")
            return False

    @staticmethod
    def eliminar_calificacion(id_calificacion: int):
        """Elimina de forma permanente un registro de la base de datos utilizando su ID."""
        try:
            # Ejecutamos la orden de eliminación filtrando por el ID
            resultado = coleccion_calificacion.delete_one({"id_calificacion": id_calificacion})

            if resultado.deleted_count > 0:
                print(f"[ÉXITO] calificacion con ID {id_calificacion} eliminado permanentemente.")
                return True
            else:
                print(f"[ALERTA] No se encontró ningún calificacion con el ID {id_calificacion} para eliminar.")
                return False

        except Exception as error:
            print(f"[ERROR EN TERMINAL] Falló la eliminación del calificacion {id_calificacion}: {error}")
            return False