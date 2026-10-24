# Importamos la instancia de la base de datos MongoDB desde el módulo 'database'
from database import db

# Referenciamos la colección específica denominada 'rol' dentro de nuestra base de datos
coleccion_rol = db["rol"]


# Definimos la clase controladora para gestionar toda la lógica de negocio de los roles
class RolController:

    @staticmethod
    def obtener_siguiente_id():
        """Calcula el siguiente ID secuencial de forma numérica para simular un campo autoincremental."""
        # Intentamos ejecutar el bloque para prevenir fallos críticos en la base de datos
        try:
            # Buscamos el último documento existente que contenga 'id_rol' ordenado de forma descendente (-1)
            ultimo_documento = coleccion_rol.find_one(
                {"id_rol": {"$exists": True}}, sort=[("id_rol", -1)]
            )

            # Verificamos si se encontró un documento previo y si contiene el campo 'id_rol'
            if ultimo_documento and "id_rol" in ultimo_documento:
                # Retornamos el valor del último ID encontrado incrementado en una unidad
                return ultimo_documento["id_rol"] + 1

            # Si la colección está completamente vacía, retornamos el número 1 como primer ID inicial
            return 1

        # Capturamos cualquier excepción o error que ocurra durante la consulta a MongoDB
        except Exception as error:
            # Imprimimos una alerta detallada en la terminal indicando el fallo técnico
            print(f"[ERROR EN TERMINAL] No se pudo calcular el siguiente ID para rol: {error}")
            # Retornamos None en caso de fallo para que el sistema lo detecte
            return None

    @staticmethod
    def crear_rol(nombre_rol: str):
        """Genera un nuevo ID e inserta un nuevo registro de rol en la base de datos."""
        # Intentamos realizar el proceso completo de inserción
        try:
            # Obtenemos el siguiente identificador numérico llamando al método estático interno
            nuevo_id = RolController.obtener_siguiente_id()

            # Validamos que el ID obtenido no sea nulo antes de proceder
            if nuevo_id is None:
                # Imprimimos una alerta en la terminal si hubo un problema generando el ID
                print("[ALERTA] Operación de creación cancelada: No se pudo generar el ID.")
                return None

            # Creamos el diccionario estructurado con los datos que se guardarán en MongoDB
            documento_rol = {"id_rol": nuevo_id, "nombre_rol": nombre_rol}

            # Insertamos el documento estructurado dentro de la colección 'rol'
            coleccion_rol.insert_one(documento_rol)

            # Imprimimos un mensaje de éxito claro en la terminal del sistema
            print(f"[EXITO] Rol creado correctamente con ID: {nuevo_id} y Nombre: {nombre_rol}")

            # Devolvemos el identificador único del rol recién creado
            return nuevo_id

        # Capturamos errores imprevistos durante la inserción en la base de datos
        except Exception as error:
            # Imprimimos la alerta con el error específico en la terminal
            print(f"[ERROR EN TERMINAL] Falló la creación del rol en MongoDB: {error}")
            return None

    @staticmethod
    def obtener_todos_los_roles():
        """Obtiene la lista completa de roles almacenados, ordenados de forma ascendente por su ID."""
        # Intentamos recuperar la lista desde la base de datos de manera segura
        try:
            # Consultamos los documentos con 'id_rol', ordenamos ascendentemente (1), ocultamos '_id' y convertimos a lista
            lista_roles = list(
                coleccion_rol.find({"id_rol": {"$exists": True}}, {"_id": 0}).sort("id_rol", 1)
            )

            # Retornamos la lista obtenida con todos los roles registrados
            return lista_roles

        # Capturamos posibles fallos de conexión o lectura
        except Exception as error:
            # Imprimimos la alerta del error en la terminal
            print(f"[ERROR EN TERMINAL] No se pudieron obtener los roles: {error}")
            # Retornamos una lista vacía para evitar que la aplicación se rompa
            return []

    @staticmethod
    def obtener_rol_por_id(id_rol: int):
        """Busca y retorna un rol específico filtrando por su identificador único."""
        # Intentamos realizar la búsqueda individual
        try:
            # Buscamos el documento que coincida con el ID y ocultamos el campo '_id' interno
            rol_encontrado = coleccion_rol.find_one({"id_rol": id_rol}, {"_id": 0})

            # Retornamos el diccionario encontrado (o None si no existe en la base de datos)
            return rol_encontrado

        # Capturamos excepciones durante la búsqueda
        except Exception as error:
            # Imprimimos la alerta en la terminal
            print(f"[ERROR EN TERMINAL] Error al buscar el rol con ID {id_rol}: {error}")
            return None

    @staticmethod
    def actualizar_rol(id_rol: int, nuevo_nombre: str):
        """Modifica el campo 'nombre_rol' de un registro existente según su ID."""
        # Intentamos ejecutar la actualización del documento
        try:
            # Ejecutamos la orden de actualización en MongoDB utilizando el operador $set
            resultado = coleccion_rol.update_one(
                {"id_rol": id_rol}, 
                {"$set": {"nombre_rol": nuevo_nombre}}
            )

            # Verificamos si se encontró y modificó al menos un documento
            if resultado.matched_count > 0:
                # Imprimimos aviso de éxito en la terminal
                print(f"[EXITO] Rol con ID {id_rol} actualizado correctamente a '{nuevo_nombre}'.")
                return True
            else:
                # Imprimimos alerta si el ID no existía
                print(f"[ALERTA] No se encontró ningún rol registrado con el ID {id_rol} para actualizar.")
                return False

        # Capturamos errores de ejecución en la base de datos
        except Exception as error:
            # Imprimimos la alerta del error técnico en la terminal
            print(f"[ERROR EN TERMINAL] Falló la actualización del rol {id_rol}: {error}")
            return False

    @staticmethod
    def eliminar_rol(id_rol: int):
        """Remueve un rol solo si ningún usuario lo tiene asignado actualmente."""
        try:
            # 1. Verificamos si hay algún usuario usando este rol
            # (Importando la colección de usuario o consultándola)
            from database import db
            coleccion_usuario = db["usuario"]
            
            usuario_existente = coleccion_usuario.find_one({"id_rol": id_rol})
            
            if usuario_existente:
                print(f"[ALERTA] No se puede eliminar el rol ID {id_rol} porque hay usuarios asignados a él.")
                return False

            # 2. Si ningún usuario lo usa, procedemos a borrarlo con seguridad
            resultado = coleccion_rol.delete_one({"id_rol": id_rol})

            if resultado.deleted_count > 0:
                print(f"[EXITO] Rol con ID {id_rol} eliminado permanentemente de la base de datos.")
                return True
            else:
                print(f"[ALERTA] No se encontró ningún rol con el ID {id_rol} para eliminar.")
                return False

        except Exception as error:
            print(f"[ERROR EN TERMINAL] Falló la eliminación del rol {id_rol}: {error}")
            return False