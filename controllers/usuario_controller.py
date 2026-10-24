from datetime import datetime
from database import db
from models.usuario_model import UsuarioModel, UsuarioUpdateModel

# Referenciamos la colección específica en la base de datos
coleccion_usuario = db["usuario"]
coleccion_rol = db["rol"]
coleccion_vehiculo = db["vehiculo"]
coleccion_viaje = db["viaje"]
coleccion_reserva = db["reserva"]
coleccion_calificacion = db["calificacion"]

class UsuarioController:
    """
    Controlador encargado de gestionar toda la lógica de negocio y operaciones CRUD
    para la entidad, comunicándose directamente con MongoDB de forma segura.
    """

    @staticmethod
    def obtener_siguiente_id():
        """Calcula y retorna el siguiente ID secuencial numérico para el registro."""
        try:
            # Buscamos el último documento registrado ordenado de forma descendente por el ID
            ultimo_documento = coleccion_usuario.find_one(
                {"id_usuario": {"$exists": True}}, sort=[("id_usuario", -1)]
            )

            # Si existe un documento previo, incrementamos su ID en uno; de lo contrario, iniciamos en 1
            if ultimo_documento and "id_usuario" in ultimo_documento:
                return ultimo_documento["id_usuario"] + 1
            return 1

        except Exception as error:
            print(f"[ERROR EN TERMINAL] No se pudo calcular el siguiente ID de usuario: {error}")
            return None

    @staticmethod
    def crear_usuario(usuario: UsuarioModel):
        """Inserta un nuevo registro en la base de datos tras validar sus datos."""
        try:
            # Obtenemos el ID correlativo único para el nuevo registro
            nuevo_id = UsuarioController.obtener_siguiente_id()
            if nuevo_id is None:
                print("[ALERTA] Operación cancelada: No se pudo generar el ID para el usuario.")
                return None

            # Convertimos el modelo Pydantic a un diccionario plano para MongoDB
            documento_usuario = usuario.model_dump()
            documento_usuario["id_usuario"] = nuevo_id

            # Si no se proporciona una fecha de registro, asignamos la fecha y hora actual
            if not documento_usuario.get("fecha_registro"):
                documento_usuario["fecha_registro"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            # Insertamos el documento estructurado en la colección correspondiente
            coleccion_usuario.insert_one(documento_usuario)
            print(f"[ÉXITO] Usuario creado correctamente con ID: {nuevo_id}")
            
            return nuevo_id

        except Exception as error:
            print(f"[ERROR EN TERMINAL] Falló la creación del usuario en MongoDB: {error}")
            return None

    @staticmethod
    def obtener_todos_los_usuarios():
        """Recupera la lista completa de todos los registros en el sistema."""
        try:
            # Consultamos todos los documentos excluyendo el campo '_id' interno de MongoDB
            lista_usuarios = list(
                coleccion_usuario.find({"id_usuario": {"$exists": True}}, {"_id": 0}).sort("id_usuario", 1)
            )
            
            # Cruzamos el id_rol con la colección rol para obtener el nombre del rol dinámicamente
            for u in lista_usuarios:
                id_rol = u.get("id_rol")
                if id_rol is not None:
                    rol_doc = coleccion_rol.find_one({"id_rol": int(id_rol)}, {"_id": 0})
                    if rol_doc and "nombre_rol" in rol_doc:
                        u["nombre_rol"] = rol_doc["nombre_rol"]
                    else:
                        u["nombre_rol"] = f"Rol #{id_rol}"
                else:
                    u["nombre_rol"] = "Sin rol"

            return lista_usuarios

        except Exception as error:
            print(f"[ERROR EN TERMINAL] No se pudieron obtener los usuarios: {error}")
            return []

    @staticmethod
    def obtener_usuario_por_id(id_usuario: int):
        """Busca y retorna un dato específico filtrando por su identificador único."""
        try:
            # Consultamos el registro que coincida con el ID indicado
            usuario_encontrado = coleccion_usuario.find_one({"id_usuario": id_usuario}, {"_id": 0})
            return usuario_encontrado

        except Exception as error:
            print(f"[ERROR EN TERMINAL] Error al buscar el usuario con ID {id_usuario}: {error}")
            return None

    @staticmethod
    def actualizar_usuario(id_usuario: int, usuario_update: UsuarioUpdateModel):
        """Actualiza de forma parcial los datos existentes según los campos enviados."""
        try:
            # Filtramos únicamente los campos que no sean nulos para evitar sobreescribir información
            datos_a_actualizar = {
                clave: valor for clave, valor in usuario_update.model_dump().items() if valor is not None
            }

            if not datos_a_actualizar:
                print(f"[ALERTA] No se proporcionaron datos válidos para actualizar al usuario {id_usuario}.")
                return False

            # Ejecutamos la actualización en la base de datos utilizando el operador $set
            resultado = coleccion_usuario.update_one(
                {"id_usuario": id_usuario}, 
                {"$set": datos_a_actualizar}
            )

            if resultado.matched_count > 0:
                print(f"[ÉXITO] Usuario con ID {id_usuario} actualizado correctamente.")
                return True
            else:
                print(f"[ALERTA] No se encontró ningún usuario con el ID {id_usuario} para actualizar.")
                return False

        except Exception as error:
            print(f"[ERROR EN TERMINAL] Falló la actualización del usuario {id_usuario}: {error}")
            return False

    @staticmethod
    def eliminar_usuario(id_usuario: int):
        """Elimina de forma permanente un registro y todos sus datos relacionados (Cascada) utilizando su ID."""
        try:
            # 1. Eliminamos los registros asociados en otras colecciones para mantener la integridad
            res_vehiculos = coleccion_vehiculo.delete_many({"id_usuario": id_usuario})
            res_viajes = coleccion_viaje.delete_many({"id_usuario": id_usuario})
            res_reservas = coleccion_reserva.delete_many({"id_usuario": id_usuario})
            res_calificaciones = coleccion_calificacion.delete_many({"id_usuario": id_usuario})

            # 2. Ejecutamos la orden de eliminación del usuario principal
            resultado = coleccion_usuario.delete_one({"id_usuario": id_usuario})

            if resultado.deleted_count > 0:
                print(f"[ÉXITO] Usuario con ID {id_usuario} y sus registros relacionados eliminados permanentemente.")
                print(f"   -> Vehículos eliminados: {res_vehiculos.deleted_count}")
                print(f"   -> Viajes eliminados: {res_viajes.deleted_count}")
                print(f"   -> Reservas eliminadas: {res_reservas.deleted_count}")
                print(f"   -> Calificaciones eliminadas: {res_calificaciones.deleted_count}")
                return True
            else:
                print(f"[ALERTA] No se encontró ningún usuario con el ID {id_usuario} para eliminar.")
                return False

        except Exception as error:
            print(f"[ERROR EN TERMINAL] Falló la eliminación en cascada del usuario {id_usuario}: {error}")
            return False