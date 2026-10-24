from datetime import datetime
from database import db
from models.ruta_model import RutaModel, RutaUpdateModel

# Referenciamos la colección específica en la base de datos
coleccion_ruta = db["ruta"]


class RutaController:
    """
    Controlador encargado de gestionar toda la lógica de negocio y operaciones CRUD
    para la entidad ruta, comunicándose directamente con MongoDB de forma segura.
    """

    @staticmethod
    def obtener_siguiente_id():
        """Calcula y retorna el siguiente ID secuencial numérico para el registro."""
        try:
            ultimo_documento = coleccion_ruta.find_one(
                {"id_ruta": {"$exists": True}}, sort=[("id_ruta", -1)]
            )
            if ultimo_documento and "id_ruta" in ultimo_documento:
                return ultimo_documento["id_ruta"] + 1
            return 1
        except Exception as error:
            print(f"[ERROR EN TERMINAL] No se pudo calcular el siguiente ID de ruta: {error}")
            return None

    @staticmethod
    def crear_ruta(ruta: RutaModel):
        """Inserta un nuevo registro en la base de datos tras validar sus datos."""
        try:
            nuevo_id = RutaController.obtener_siguiente_id()
            if nuevo_id is None:
                print("[ALERTA] Operación cancelada: No se pudo generar el ID para la ruta.")
                return None

            documento_ruta = ruta.model_dump()
            documento_ruta["id_ruta"] = nuevo_id

            if not documento_ruta.get("fecha_registro"):
                documento_ruta["fecha_registro"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            coleccion_ruta.insert_one(documento_ruta)
            print(f"[ÉXITO] Ruta creada correctamente con ID: {nuevo_id}")
            return nuevo_id

        except Exception as error:
            print(f"[ERROR EN TERMINAL] Falló la creación de la ruta en MongoDB: {error}")
            return None

    @staticmethod
    def obtener_todas_las_rutas():
        """Recupera la lista completa de todas las rutas enriquecidas con el nombre del usuario."""
        try:
            lista_rutas = list(
                coleccion_ruta.find({"id_ruta": {"$exists": True}}, {"_id": 0}).sort("id_ruta", 1)
            )

            # Enriquecemos cada ruta resolviendo el nombre del usuario creador
            for ruta in lista_rutas:
                id_usu = ruta.get("id_usuario")
                if id_usu is not None:
                    usuario_doc = db["usuario"].find_one({"id_usuario": id_usu}, {"_id": 0, "nombre": 1, "apellido_paterno": 1})
                    if usuario_doc:
                        ruta["nombre_usuario"] = f"{usuario_doc.get('nombre', '')} {usuario_doc.get('apellido_paterno', '')}".strip()
                    else:
                        ruta["nombre_usuario"] = "Usuario Desconocido"
                else:
                    ruta["nombre_usuario"] = "Sin Asignar"

            return lista_rutas

        except Exception as error:
            print(f"[ERROR EN TERMINAL] No se pudieron obtener las rutas: {error}")
            return []

    @staticmethod
    def obtener_ruta_por_id(id_ruta: int):
        """Busca y retorna una ruta específica filtrando por su identificador único."""
        try:
            ruta_encontrada = coleccion_ruta.find_one({"id_ruta": id_ruta}, {"_id": 0})
            return ruta_encontrada
        except Exception as error:
            print(f"[ERROR EN TERMINAL] Error al buscar la ruta con ID {id_ruta}: {error}")
            return None

    @staticmethod
    def actualizar_ruta(id_ruta: int, ruta_update: RutaUpdateModel):
        """Actualiza de forma parcial los datos existentes según los campos enviados."""
        try:
            datos_a_actualizar = {
                clave: valor for clave, valor in ruta_update.model_dump().items() if valor is not None
            }

            if not datos_a_actualizar:
                print(f"[ALERTA] No se proporcionaron datos válidos para actualizar la ruta {id_ruta}.")
                return False

            resultado = coleccion_ruta.update_one(
                {"id_ruta": id_ruta}, 
                {"$set": datos_a_actualizar}
            )

            if resultado.matched_count > 0:
                print(f"[ÉXITO] Ruta con ID {id_ruta} actualizada correctamente.")
                return True
            else:
                print(f"[ALERTA] No se encontró ninguna ruta con el ID {id_ruta} para actualizar.")
                return False

        except Exception as error:
            print(f"[ERROR EN TERMINAL] Falló la actualización de la ruta {id_ruta}: {error}")
            return False

    @staticmethod
    def eliminar_ruta(id_ruta: int):
        """Elimina de forma permanente un registro de la base de datos utilizando su ID."""
        try:
            resultado = coleccion_ruta.delete_one({"id_ruta": id_ruta})

            if resultado.deleted_count > 0:
                print(f"[ÉXITO] Ruta con ID {id_ruta} eliminada permanentemente.")
                return True
            else:
                print(f"[ALERTA] No se encontró ninguna ruta con el ID {id_ruta} para eliminar.")
                return False

        except Exception as error:
            print(f"[ERROR EN TERMINAL] Falló la eliminación de la ruta {id_ruta}: {error}")
            return False