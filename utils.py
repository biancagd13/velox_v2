def obtener_siguiente_id(coleccion, campo_id):
    """
    Obtiene el siguiente ID autoincremental buscando el valor máximo actual.
    """
    ultimo = coleccion.find_one(sort=[(campo_id, -1)])
    if ultimo and campo_id in ultimo:
        return ultimo[campo_id] + 1
    return 1