# Importamos BaseModel para definir la estructura de datos y Field para la configuración de campos
from pydantic import BaseModel, Field

# Importamos los Enums desde el archivo central que creamos
from models.enums import EstadoReserva

# Definición de la clase del modelo para la entidad 'reserva' (Creación)
class ReservaModel(BaseModel):
    # Identificador único autoincrementable del reserva (opcional al crear)
    id_reserva: int | None = Field(default=None)

    # Campos obligatorios
    estado: EstadoReserva = Field(default=EstadoReserva.PENDIENTE)
    fecha_solicitud: str = Field(...)

    id_usuario: int = Field(...)
    id_viaje: int = Field(...)
    
# Definición de la clase para la actualización de datos 
class ReservaUpdateModel(BaseModel):
    # Todos los campos son opcionales para permitir actualizaciones parciales
    estado: EstadoReserva | None = None
    
    id_usuario: int | None = None
    id_viaje: int | None = None