from datetime import datetime
from database import db
from models.viaje_model import ViajeModel, ViajeUpdateModel

# Referenciamos la colección específica en la base de datos
coleccion_viaje = db["viaje"]


class ViajeController:
    """
    Controlador encargado de gestionar toda la lógica de negocio y operaciones CRUD
    para la entidad viaje, comunicándose directamente con MongoDB de forma segura.
    """

    @staticmethod
    def obtener_siguiente_id():
        """Calcula y retorna el siguiente ID secuencial numérico para el registro."""
        try:
            ultimo_documento = coleccion_viaje.find_one(
                {"id_viaje": {"$exists": True}}, sort=[("id_viaje", -1)]
            )
            if ultimo_documento and "id_viaje" in ultimo_documento:
                return ultimo_documento["id_viaje"] + 1
            return 1
        except Exception as error:
            print(f"[ERROR EN TERMINAL] No se pudo calcular el siguiente ID de viaje: {error}")
            return None

    @staticmethod
    def crear_viaje(viaje: ViajeModel):
        """Inserta un nuevo registro en la base de datos tras validar sus datos."""
        try:
            nuevo_id = ViajeController.obtener_siguiente_id()
            if nuevo_id is None:
                print("[ALERTA] Operación cancelada: No se pudo generar el ID para el viaje.")
                return None

            documento_viaje = viaje.model_dump()
            documento_viaje["id_viaje"] = nuevo_id

            # Si el estado es un Enum de Pydantic, lo convertimos a su valor de cadena (string) para MongoDB
            if hasattr(documento_viaje.get("estado"), "value"):
                documento_viaje["estado"] = documento_viaje["estado"].value

            if not documento_viaje.get("fecha_registro"):
                documento_viaje["fecha_registro"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            coleccion_viaje.insert_one(documento_viaje)
            print(f"[ÉXITO] Viaje creado correctamente con ID: {nuevo_id}")
            return nuevo_id

        except Exception as error:
            print(f"[ERROR EN TERMINAL] Falló la creación del viaje en MongoDB: {error}")
            return None

    @staticmethod
    def obtener_todos_los_viajes():
        """Recupera la lista completa de todos los viajes enriquecidos con la ruta y el conductor."""
        try:
            lista_viajes = list(
                coleccion_viaje.find({"id_viaje": {"$exists": True}}, {"_id": 0}).sort("id_viaje", 1)
            )

            # Enriquecemos cada viaje cruzando con ruta y usuario
            for viaje in lista_viajes:
                # 1. Resolver Ruta Asociada
                id_rt = viaje.get("id_ruta")
                if id_rt is not None:
                    ruta_doc = db["ruta"].find_one({"id_ruta": id_rt}, {"_id": 0})
                    if ruta_doc:
                        origen = ruta_doc.get("origen", "Origen")
                        destino = ruta_doc.get("destino", "Destino")
                        viaje["descripcion_ruta"] = f"Ruta #{id_rt}: {origen} ➔ {destino}"
                    else:
                        viaje["descripcion_ruta"] = f"Ruta #{id_rt}"
                else:
                    viaje["descripcion_ruta"] = "Sin Ruta Asignada"

                # 2. Resolver Conductor (Usuario)
                id_usu = viaje.get("id_usuario")
                if id_usu is not None:
                    usuario_doc = db["usuario"].find_one({"id_usuario": id_usu}, {"_id": 0, "nombre": 1, "apellido_paterno": 1})
                    if usuario_doc:
                        viaje["nombre_conductor"] = f"{usuario_doc.get('nombre', '')} {usuario_doc.get('apellido_paterno', '')}".strip()
                    else:
                        viaje["nombre_conductor"] = "Conductor Desconocido"
                else:
                    viaje["nombre_conductor"] = "Sin Asignar"

            return lista_viajes

        except Exception as error:
            print(f"[ERROR EN TERMINAL] No se pudieron obtener los viajes: {error}")
            return []

    @staticmethod
    def obtener_viaje_por_id(id_viaje: int):
        """Busca y retorna un viaje específico filtrando por su identificador único."""
        try:
            viaje_encontrado = coleccion_viaje.find_one({"id_viaje": id_viaje}, {"_id": 0})
            return viaje_encontrado
        except Exception as error:
            print(f"[ERROR EN TERMINAL] Error al buscar el viaje con ID {id_viaje}: {error}")
            return None

    @staticmethod
    def actualizar_viaje(id_viaje: int, viaje_update: ViajeUpdateModel):
        """Actualiza de forma parcial los datos existentes según los campos enviados."""
        try:
            datos_a_actualizar = {
                clave: valor for clave, valor in viaje_update.model_dump().items() if valor is not None
            }

            # Convertir Enum a string si viene en la actualización
            if "estado" in datos_a_actualizar and hasattr(datos_a_actualizar["estado"], "value"):
                datos_a_actualizar["estado"] = datos_a_actualizar["estado"].value

            if not datos_a_actualizar:
                print(f"[ALERTA] No se proporcionaron datos válidos para actualizar al viaje {id_viaje}.")
                return False

            resultado = coleccion_viaje.update_one(
                {"id_viaje": id_viaje}, 
                {"$set": datos_a_actualizar}
            )

            if resultado.matched_count > 0:
                print(f"[ÉXITO] Viaje con ID {id_viaje} actualizado correctamente.")
                return True
            else:
                print(f"[ALERTA] No se encontró ningún viaje con el ID {id_viaje} para actualizar.")
                return False

        except Exception as error:
            print(f"[ERROR EN TERMINAL] Falló la actualización del viaje {id_viaje}: {error}")
            return False

    @staticmethod
    def eliminar_viaje(id_viaje: int):
        """Elimina de forma permanente un registro de la base de datos utilizando su ID."""
        try:
            resultado = coleccion_viaje.delete_one({"id_viaje": id_viaje})

            if resultado.deleted_count > 0:
                print(f"[ÉXITO] Viaje con ID {id_viaje} eliminado permanentemente.")
                return True
            else:
                print(f"[ALERTA] No se encontró ningún viaje con el ID {id_viaje} para eliminar.")
                return False

        except Exception as error:
            print(f"[ERROR EN TERMINAL] Falló la eliminación del viaje {id_viaje}: {error}")
            return False