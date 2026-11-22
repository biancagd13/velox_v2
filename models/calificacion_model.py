from pydantic import BaseModel, Field


class CalificacionModel(BaseModel):
    id_calificacion: int | None = Field(default=None)
    puntuacion: int = Field(...)
    comentario: str = Field(...)
    id_usuario: int = Field(...)
    id_viaje: int = Field(...)


class CalificacionUpdateModel(BaseModel):
    puntuacion: int | None = None
    comentario: str | None = None
    id_usuario: int | None = None
    id_viaje: int | None = None