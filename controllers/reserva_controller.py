from datetime import datetime
from database import db
from models.reserva_model import ReservaModel, ReservaUpdateModel

# Referenciamos la colección específica en la base de datos
coleccion_reserva = db["reserva"]


class ReservaController:
    """
    Controlador encargado de gestionar toda la lógica de negocio y operaciones CRUD
    para la entidad reserva, comunicándose directamente con MongoDB de forma segura.
    """

    @staticmethod
    def obtener_siguiente_id():
        """Calcula y retorna el siguiente ID secuencial numérico para el registro."""
        try:
            ultimo_documento = coleccion_reserva.find_one(
                {"id_reserva": {"$exists": True}}, sort=[("id_reserva", -1)]
            )
            if ultimo_documento and "id_reserva" in ultimo_documento:
                return ultimo_documento["id_reserva"] + 1
            return 1
        except Exception as error:
            print(f"[ERROR EN TERMINAL] No se pudo calcular el siguiente ID de reserva: {error}")
            return None

    @staticmethod
    def crear_reserva(reserva: ReservaModel):
        """Inserta un nuevo registro en la base de datos tras validar sus datos."""
        try:
            nuevo_id = ReservaController.obtener_siguiente_id()
            if nuevo_id is None:
                print("[ALERTA] Operación cancelada: No se pudo generar el ID para la reserva.")
                return None

            documento_reserva = reserva.model_dump()
            documento_reserva["id_reserva"] = nuevo_id

            # Si el estado es un Enum de Pydantic, lo convertimos a su valor de cadena (string) para MongoDB
            if hasattr(documento_reserva.get("estado"), "value"):
                documento_reserva["estado"] = documento_reserva["estado"].value

            if not documento_reserva.get("fecha_solicitud"):
                documento_reserva["fecha_solicitud"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            coleccion_reserva.insert_one(documento_reserva)
            print(f"[ÉXITO] Reserva creada correctamente con ID: {nuevo_id}")
            return nuevo_id

        except Exception as error:
            print(f"[ERROR EN TERMINAL] Falló la creación de la reserva en MongoDB: {error}")
            return None

    @staticmethod
    def obtener_todas_las_reservas():
        """Recupera la lista completa de todas las reservas enriquecidas con el usuario y detalles del viaje."""
        try:
            lista_reservas = list(
                coleccion_reserva.find({"id_reserva": {"$exists": True}}, {"_id": 0}).sort("id_reserva", 1)
            )

            # Enriquecemos cada reserva cruzando con usuario y viaje
            for reserva in lista_reservas:
                # 1. Resolver Nombre del Usuario Pasajero
                id_usu = reserva.get("id_usuario")
                if id_usu is not None:
                    usuario_doc = db["usuario"].find_one({"id_usuario": id_usu}, {"_id": 0, "nombre": 1, "apellido_paterno": 1})
                    if usuario_doc:
                        reserva["nombre_usuario"] = f"{usuario_doc.get('nombre', '')} {usuario_doc.get('apellido_paterno', '')}".strip()
                    else:
                        reserva["nombre_usuario"] = "Usuario Desconocido"
                else:
                    reserva["nombre_usuario"] = "Sin Asignar"

                # 2. Resolver Ruta del Viaje Asociado
                id_vj = reserva.get("id_viaje")
                if id_vj is not None:
                    viaje_doc = db["viaje"].find_one({"id_viaje": id_vj}, {"_id": 0})
                    if viaje_doc:
                        origen = viaje_doc.get("origen") or viaje_doc.get("punto_partida")
                        destino = viaje_doc.get("destino") or viaje_doc.get("punto_llegada")
                        
                        if (not origen or not destino) and "id_ruta" in viaje_doc:
                            ruta_doc = db["ruta"].find_one({"id_ruta": viaje_doc["id_ruta"]}, {"_id": 0})
                            if ruta_doc:
                                origen = origen or ruta_doc.get("origen") or ruta_doc.get("punto_partida", "Origen")
                                destino = destino or ruta_doc.get("destino") or ruta_doc.get("punto_llegada", "Destino")
                        
                        origen = origen or "Origen"
                        destino = destino or "Destino"
                        reserva["descripcion_viaje"] = f"Viaje #{id_vj}: {origen} ➔ {destino}"
                    else:
                        reserva["descripcion_viaje"] = f"Viaje #{id_vj}"
                else:
                    reserva["descripcion_viaje"] = "Sin Viaje Asignado"

            return lista_reservas

        except Exception as error:
            print(f"[ERROR EN TERMINAL] No se pudieron obtener las reservas: {error}")
            return []

    @staticmethod
    def obtener_reserva_por_id(id_reserva: int):
        """Busca y retorna una reserva específica filtrando por su identificador único."""
        try:
            reserva_encontrada = coleccion_reserva.find_one({"id_reserva": id_reserva}, {"_id": 0})
            return reserva_encontrada
        except Exception as error:
            print(f"[ERROR EN TERMINAL] Error al buscar la reserva con ID {id_reserva}: {error}")
            return None

    @staticmethod
    def actualizar_reserva(id_reserva: int, reserva_update: ReservaUpdateModel):
        """Actualiza de forma parcial los datos existentes según los campos enviados."""
        try:
            datos_a_actualizar = {
                clave: valor for clave, valor in reserva_update.model_dump().items() if valor is not None
            }

            # Convertir Enum a string si viene en la actualización
            if "estado" in datos_a_actualizar and hasattr(datos_a_actualizar["estado"], "value"):
                datos_a_actualizar["estado"] = datos_a_actualizar["estado"].value

            if not datos_a_actualizar:
                print(f"[ALERTA] No se proporcionaron datos válidos para actualizar la reserva {id_reserva}.")
                return False

            resultado = coleccion_reserva.update_one(
                {"id_reserva": id_reserva}, 
                {"$set": datos_a_actualizar}
            )

            if resultado.matched_count > 0:
                print(f"[ÉXITO] Reserva con ID {id_reserva} actualizada correctamente.")
                return True
            else:
                print(f"[ALERTA] No se encontró ninguna reserva con el ID {id_reserva} para actualizar.")
                return False

        except Exception as error:
            print(f"[ERROR EN TERMINAL] Falló la actualización de la reserva {id_reserva}: {error}")
            return False

    @staticmethod
    def eliminar_reserva(id_reserva: int):
        """Elimina de forma permanente un registro de la base de datos utilizando su ID."""
        try:
            resultado = coleccion_reserva.delete_one({"id_reserva": id_reserva})

            if resultado.deleted_count > 0:
                print(f"[ÉXITO] Reserva con ID {id_reserva} eliminada permanentemente.")
                return True
            else:
                print(f"[ALERTA] No se encontró ninguna reserva con el ID {id_reserva} para eliminar.")
                return False

        except Exception as error:
            print(f"[ERROR EN TERMINAL] Falló la eliminación de la reserva {id_reserva}: {error}")
            return False