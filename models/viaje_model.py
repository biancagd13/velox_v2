# Importamos BaseModel para definir la estructura de datos y Field para la configuración de campos
from pydantic import BaseModel, Field


# Definición de la clase del modelo para la entidad 'Viaje' (Creación)
class ViajeModel(BaseModel):
    # Identificador único autoincrementable del Viaje (opcional al crear)
    id_viaje: int | None = Field(default=None)

    # Campos obligatorios
    fecha: str = Field(...)
    asientos_disponibles: int = Field(...)
    costo: int = Field(...)
    estado: str = Field(default="pendiente")

    id_ruta: int = Field(...)
    id_usuario: int = Field(...)
    
# Definición de la clase para la actualización de datos 
class ViajeUpdateModel(BaseModel):
    # Todos los campos son opcionales para permitir actualizaciones parciales
    fecha: str | None = None
    asientos_disponibles: int | None = None
    costo: int | None = None
    estado: str | None = None

    id_ruta: int | None = None
    id_usuario: int | None = None