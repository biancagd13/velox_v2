# Importamos BaseModel para definir la estructura de datos y Field para la configuración de los campos
from pydantic import BaseModel, Field


# Definición de la clase del modelo para la entidad 'Rol'
class RolModel(BaseModel):
  # Campo identificador del rol: es de tipo entero y opcional (None por defecto al crear uno nuevo)
  id_rol: int | None = Field(default=None)

  # Campo para almacenar el nombre del rol: es de tipo texto y de carácter obligatorio
  nombre_rol: str