# Importamos BaseModel para definir la estructura de datos y Field para la configuración de campos
from pydantic import BaseModel, Field


# Definición de la clase del modelo para la entidad 'Ruta' (Creación)
class RutaModel(BaseModel):
    # Identificador único autoincrementable del Ruta (opcional al crear)
    id_ruta: int | None = Field(default=None)

    # Campos obligatorios
    origen: str = Field(...)
    destino: str = Field(...)
    hora_salida: str = Field(...)

    id_usuario: int = Field(...)
    
# Definición de la clase para la actualización de datos 
class RutaUpdateModel(BaseModel):
    # Todos los campos son opcionales para permitir actualizaciones parciales
    origen: str | None = None
    destino: str | None = None
    hora_salida: str | None = None
    
    id_usuario: int | None = None
