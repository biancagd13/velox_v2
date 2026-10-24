from enum import Enum

# Estados para los viajes
class EstadoViaje(str, Enum):
    PENDIENTE = "pendiente"
    ACEPTADO = "aceptado"
    CANCELADO = "cancelado"
    FINALIZADO = "finalizado"

# Estados para las reservas 

class EstadoReserva(str, Enum):
    PENDIENTE = "pendiente"
    CONFIRMADA = "confirmada"
    CANCELADA = "cancelada"