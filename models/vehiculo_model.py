# Importamos BaseModel para definir la estructura de datos y Field para la configuración de campos
from pydantic import BaseModel, Field


# Definición de la clase del modelo para la entidad 'vehiculo' (Creación)
class VehiculoModel(BaseModel):
    # Identificador único autoincrementable del vehiculo (opcional al crear)
    id_vehiculo: int | None = Field(default=None)

    # Campos obligatorios
    placa: str = Field(...)
    nombre: str = Field(...)
    modelo: str = Field(...)
    color: str = Field(...)
    capacidad_asientos: int = Field(...)

    id_usuario: int = Field(...)
    
# Definición de la clase para la actualización de datos 
class VehiculoUpdateModel(BaseModel):
    # Todos los campos son opcionales para permitir actualizaciones parciales
    placa: str | None = None
    nombre: str | None = None
    modelo: str | None = None
    color: str | None = None
    capacidad_asientos: int | None = None

    id_usuario: int | None = None