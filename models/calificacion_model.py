# Importamos BaseModel para definir la estructura de datos y Field para la configuración de campos
from pydantic import BaseModel, Field


# Definición de la clase del modelo para la entidad 'Calificacion' (Creación)
class CalificacionModel(BaseModel):
    # Identificador único autoincrementable del Calificacion (opcional al crear)
    id_calificacion: int | None = Field(default=None)

    # Campos obligatorios
    puntuacion: int = Field(...)
    comentario: str = Field(...)

    id_usuario: int = Field(...)
    id_viaje: int = Field(...)
    
# Definición de la clase para la actualización de datos 
class CalificacionUpdateModel(BaseModel):
    # Todos los campos son opcionales para permitir actualizaciones parciales
    puntuacion: int | None = None
    comentario: str | None = None
    
    id_usuario: int | None = None
    id_viaje: int | None = None