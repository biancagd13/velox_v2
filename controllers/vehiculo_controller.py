from datetime import datetime
from database import db
from models.vehiculo_model import VehiculoModel, VehiculoUpdateModel

# Referenciamos la colección específica en la base de datos
coleccion_vehiculo = db["vehiculo"]


class VehiculoController:
    """
    Controlador encargado de gestionar toda la lógica de negocio y operaciones CRUD
    para la entidad vehículo, comunicándose directamente con MongoDB de forma segura.
    """

    @staticmethod
    def obtener_siguiente_id():
        """Calcula y retorna el siguiente ID secuencial numérico para el registro."""
        try:
            ultimo_documento = coleccion_vehiculo.find_one(
                {"id_vehiculo": {"$exists": True}}, sort=[("id_vehiculo", -1)]
            )
            if ultimo_documento and "id_vehiculo" in ultimo_documento:
                return ultimo_documento["id_vehiculo"] + 1
            return 1
        except Exception as error:
            print(f"[ERROR EN TERMINAL] No se pudo calcular el siguiente ID de vehículo: {error}")
            return None

    @staticmethod
    def crear_vehiculo(vehiculo: VehiculoModel):
        """Inserta un nuevo registro en la base de datos tras validar sus datos."""
        try:
            nuevo_id = VehiculoController.obtener_siguiente_id()
            if nuevo_id is None:
                print("[ALERTA] Operación cancelada: No se pudo generar el ID para el vehículo.")
                return None

            documento_vehiculo = vehiculo.model_dump()
            documento_vehiculo["id_vehiculo"] = nuevo_id

            if not documento_vehiculo.get("fecha_registro"):
                documento_vehiculo["fecha_registro"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            coleccion_vehiculo.insert_one(documento_vehiculo)
            print(f"[ÉXITO] Vehículo creado correctamente con ID: {nuevo_id}")
            return nuevo_id

        except Exception as error:
            print(f"[ERROR EN TERMINAL] Falló la creación del vehículo en MongoDB: {error}")
            return None

    @staticmethod
    def obtener_todos_los_vehiculos():
        """Recupera la lista completa de todos los vehículos enriquecidos con el nombre del propietario."""
        try:
            lista_vehiculos = list(
                coleccion_vehiculo.find({"id_vehiculo": {"$exists": True}}, {"_id": 0}).sort("id_vehiculo", 1)
            )

            # Enriquecemos cada vehículo resolviendo el nombre del propietario
            for vehiculo in lista_vehiculos:
                id_usu = vehiculo.get("id_usuario")
                if id_usu is not None:
                    usuario_doc = db["usuario"].find_one({"id_usuario": id_usu}, {"_id": 0, "nombre": 1, "apellido_paterno": 1})
                    if usuario_doc:
                        vehiculo["nombre_propietario"] = f"{usuario_doc.get('nombre', '')} {usuario_doc.get('apellido_paterno', '')}".strip()
                    else:
                        vehiculo["nombre_propietario"] = "Propietario Desconocido"
                else:
                    vehiculo["nombre_propietario"] = "Sin Asignar"

            return lista_vehiculos

        except Exception as error:
            print(f"[ERROR EN TERMINAL] No se pudieron obtener los vehículos: {error}")
            return []

    @staticmethod
    def obtener_vehiculo_por_id(id_vehiculo: int):
        """Busca y retorna un vehículo específico filtrando por su identificador único."""
        try:
            vehiculo_encontrado = coleccion_vehiculo.find_one({"id_vehiculo": id_vehiculo}, {"_id": 0})
            return vehiculo_encontrado
        except Exception as error:
            print(f"[ERROR EN TERMINAL] Error al buscar el vehículo con ID {id_vehiculo}: {error}")
            return None

    @staticmethod
    def actualizar_vehiculo(id_vehiculo: int, vehiculo_update: VehiculoUpdateModel):
        """Actualiza de forma parcial los datos existentes según los campos enviados."""
        try:
            datos_a_actualizar = {
                clave: valor for clave, valor in vehiculo_update.model_dump().items() if valor is not None
            }

            if not datos_a_actualizar:
                print(f"[ALERTA] No se proporcionaron datos válidos para actualizar al vehículo {id_vehiculo}.")
                return False

            resultado = coleccion_vehiculo.update_one(
                {"id_vehiculo": id_vehiculo}, 
                {"$set": datos_a_actualizar}
            )

            if resultado.matched_count > 0:
                print(f"[ÉXITO] Vehículo con ID {id_vehiculo} actualizado correctamente.")
                return True
            else:
                print(f"[ALERTA] No se encontró ningún vehículo con el ID {id_vehiculo} para actualizar.")
                return False

        except Exception as error:
            print(f"[ERROR EN TERMINAL] Falló la actualización del vehículo {id_vehiculo}: {error}")
            return False

    @staticmethod
    def eliminar_vehiculo(id_vehiculo: int):
        """Elimina de forma permanente un registro de la base de datos utilizando su ID."""
        try:
            resultado = coleccion_vehiculo.delete_one({"id_vehiculo": id_vehiculo})

            if resultado.deleted_count > 0:
                print(f"[ÉXITO] Vehículo con ID {id_vehiculo} eliminado permanentemente.")
                return True
            else:
                print(f"[ALERTA] No se encontró ningún vehículo con el ID {id_vehiculo} para eliminar.")
                return False

        except Exception as error:
            print(f"[ERROR EN TERMINAL] Falló la eliminación del vehículo {id_vehiculo}: {error}")
            return False