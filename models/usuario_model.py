# Importamos BaseModel para definir la estructura de datos y Field para la configuración de campos
from pydantic import BaseModel, Field


# Definición de la clase del modelo para la entidad 'Usuario' (Creación)
class UsuarioModel(BaseModel):
    # Identificador único autoincrementable del usuario (opcional al crear)
    id_usuario: int | None = Field(default=None)

    # Campos obligatorios

    matricula: str = Field(...)
    nombre: str = Field(...)
    apellido_paterno: str = Field(...)
    correo_institucional: str = Field(...)
    contrasena: str = Field(...)
    apellido_materno: str | None = Field(...)
    telefono: str | None = Field(...)
    fecha_registro: str | None = Field(...)

    id_rol: int = Field(...)


# Definición de la clase para la actualización de datos 
class UsuarioUpdateModel(BaseModel):
    # Todos los campos son opcionales para permitir actualizaciones parciales
    matricula: str | None = None
    nombre: str | None = None
    apellido_paterno: str | None = None
    apellido_materno: str | None = None
    correo_institucional: str | None = None
    telefono: str | None = None
    contrasena: str | None = None
    
    id_rol: int | None = None