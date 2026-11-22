from datetime import datetime
from database import db
from models.calificacion_model import CalificacionModel, CalificacionUpdateModel

# Referenciamos la colección específica en la base de datos
coleccion_calificacion = db["calificacion"]


def _a_entero_o_none(valor):
    """Convierte a entero si el valor es numérico válido; de lo contrario retorna None."""
    if valor is None:
        return None
    texto = str(valor).strip()
    if texto.isdigit():
        return int(texto)
    return None


def _resolver_relaciones_viaje(id_viaje: int | None):
    """Busca en 'viaje' y 'ruta' para obtener id_ruta e id_conductor asociados al viaje."""
    id_vj = _a_entero_o_none(id_viaje)
    if id_vj is None:
        return None, None

    viaje_doc = db["viaje"].find_one({"id_viaje": id_vj}, {"_id": 0})
    id_ruta = _a_entero_o_none(viaje_doc.get("id_ruta")) if viaje_doc else None
    ruta_doc = db["ruta"].find_one({"id_ruta": id_ruta}, {"_id": 0}) if id_ruta is not None else None

    id_conductor = None
    for doc_ref in (viaje_doc, ruta_doc):
        if doc_ref:
            for campo in ("id_conductor", "conductor_id", "id_usuario", "usuario_id"):
                val_dueno = _a_entero_o_none(doc_ref.get(campo))
                if val_dueno is not None:
                    id_conductor = val_dueno
                    break
        if id_conductor is not None:
            break

    return id_ruta, id_conductor


class CalificacionController:
    """
    Controlador encargado de gestionar toda la lógica de negocio y operaciones CRUD
    para la entidad, comunicándose directamente con MongoDB de forma segura.
    """

    @staticmethod
    def obtener_siguiente_id():
        """Calcula y retorna el siguiente ID secuencial numérico para el registro."""
        try:
            ultimo_documento = coleccion_calificacion.find_one(
                {"id_calificacion": {"$exists": True}}, sort=[("id_calificacion", -1)]
            )

            if ultimo_documento and "id_calificacion" in ultimo_documento:
                return int(ultimo_documento["id_calificacion"]) + 1
            return 1

        except Exception as error:
            print(f"[ERROR EN TERMINAL] No se pudo calcular el siguiente ID de calificacion: {error}")
            return None

    @staticmethod
    def crear_calificacion(calificacion: CalificacionModel):
        """Inserta un nuevo registro en la base de datos tras validar sus datos y vincular su viaje/ruta."""
        try:
            nuevo_id = CalificacionController.obtener_siguiente_id()
            if nuevo_id is None:
                print("[ALERTA] Operación cancelada: No se pudo generar el ID para el calificacion.")
                return None

            documento_calificacion = calificacion.model_dump()
            documento_calificacion["id_calificacion"] = nuevo_id

            if not documento_calificacion.get("fecha_registro"):
                documento_calificacion["fecha_registro"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            id_ruta, id_conductor = _resolver_relaciones_viaje(documento_calificacion.get("id_viaje"))
            if id_ruta is not None:
                documento_calificacion["id_ruta"] = id_ruta
            if id_conductor is not None:
                documento_calificacion["id_conductor"] = id_conductor

            coleccion_calificacion.insert_one(documento_calificacion)
            print(f"[ÉXITO] calificacion creada correctamente con ID: {nuevo_id}")

            return nuevo_id

        except Exception as error:
            print(f"[ERROR EN TERMINAL] Falló la creación del calificacion en MongoDB: {error}")
            return None

    @staticmethod
    def obtener_todas_las_calificaciones():
        """Recupera la lista completa de calificaciones enriquecidas con usuario, viaje, ruta y dueño del viaje."""
        try:
            calificaciones = list(
                coleccion_calificacion.find({"id_calificacion": {"$exists": True}}, {"_id": 0}).sort("id_calificacion", 1)
            )

            for calificacion in calificaciones:
                # 1. Resolver Nombre del Usuario (solicitante/pasajero)
                id_usu = _a_entero_o_none(calificacion.get("id_usuario") or calificacion.get("id_pasajero"))
                if id_usu is not None:
                    usuario_doc = db["usuario"].find_one(
                        {"id_usuario": id_usu},
                        {"_id": 0, "nombre": 1, "apellido_paterno": 1},
                    )
                    if usuario_doc:
                        calificacion["nombre_usuario"] = (
                            f"{usuario_doc.get('nombre', '')} {usuario_doc.get('apellido_paterno', '')}".strip()
                            or "Usuario Desconocido"
                        )
                    else:
                        calificacion["nombre_usuario"] = "Usuario Desconocido"
                else:
                    calificacion["nombre_usuario"] = "Sin Asignar"

                # 2. Resolver Viaje, Ruta y Dueño/Conductor asociado al viaje
                id_vj = _a_entero_o_none(calificacion.get("id_viaje") or calificacion.get("viaje_id"))
                id_rt = _a_entero_o_none(calificacion.get("id_ruta") or calificacion.get("ruta_id"))

                viaje_doc = None
                if id_vj is not None:
                    viaje_doc = db["viaje"].find_one({"id_viaje": id_vj}, {"_id": 0})
                    if viaje_doc and id_rt is None:
                        id_rt = _a_entero_o_none(viaje_doc.get("id_ruta"))

                ruta_doc = db["ruta"].find_one({"id_ruta": id_rt}, {"_id": 0}) if id_rt is not None else None

                dueno_viaje = _a_entero_o_none(calificacion.get("id_conductor") or calificacion.get("conductor_id"))
                if dueno_viaje is None:
                    for doc_ref in (viaje_doc, ruta_doc):
                        if doc_ref:
                            for campo in ("id_conductor", "conductor_id", "id_usuario", "usuario_id"):
                                val_dueno = _a_entero_o_none(doc_ref.get(campo))
                                if val_dueno is not None:
                                    dueno_viaje = val_dueno
                                    break
                        if dueno_viaje is not None:
                            break

                if dueno_viaje is not None:
                    calificacion["id_conductor"] = dueno_viaje
                    calificacion["id_conductor_viaje"] = dueno_viaje
                if id_rt is not None:
                    calificacion["id_ruta"] = id_rt

                if id_vj is not None:
                    origen = (viaje_doc or {}).get("origen") or (viaje_doc or {}).get("punto_partida")
                    destino = (viaje_doc or {}).get("destino") or (viaje_doc or {}).get("punto_llegada")

                    if (not origen or not destino) and ruta_doc:
                        origen = origen or ruta_doc.get("origen") or ruta_doc.get("punto_partida", "Origen")
                        destino = destino or ruta_doc.get("destino") or ruta_doc.get("punto_llegada", "Destino")

                    if origen and destino:
                        calificacion["descripcion_viaje"] = f"Viaje #{id_vj}: {origen} ➔ {destino}"
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
            calificacion_encontrado = coleccion_calificacion.find_one(
                {"id_calificacion": id_calificacion}, {"_id": 0}
            )
            return calificacion_encontrado

        except Exception as error:
            print(f"[ERROR EN TERMINAL] Error al buscar la calificacion con ID {id_calificacion}: {error}")
            return None

    @staticmethod
    def actualizar_calificacion(id_calificacion: int, calificacion_update: CalificacionUpdateModel):
        """Actualiza de forma parcial los datos existentes según los campos enviados."""
        try:
            datos_a_actualizar = {
                clave: valor for clave, valor in calificacion_update.model_dump().items() if valor is not None
            }

            if not datos_a_actualizar:
                print(f"[ALERTA] No se proporcionaron datos válidos para actualizar al calificacion {id_calificacion}.")
                return False

            if "id_viaje" in datos_a_actualizar:
                id_ruta, id_conductor = _resolver_relaciones_viaje(datos_a_actualizar["id_viaje"])
                if id_ruta is not None:
                    datos_a_actualizar["id_ruta"] = id_ruta
                if id_conductor is not None:
                    datos_a_actualizar["id_conductor"] = id_conductor

            resultado = coleccion_calificacion.update_one(
                {"id_calificacion": id_calificacion},
                {"$set": datos_a_actualizar},
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