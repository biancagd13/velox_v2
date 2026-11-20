from database import db

def poblar_viajes_con_id(cantidad=50):
    # Obtenemos el último id_viaje existente para continuar la secuencia
    ultimo = db["viaje"].find_one({"id_viaje": {"$exists": True}}, sort=[("id_viaje", -1)])
    siguiente_id = (ultimo["id_viaje"] + 1) if ultimo and "id_viaje" in ultimo else 1
    
    estados = ["en_curso", "finalizado", "activo", "pendiente", "cancelado"]
    rutas = [1, 2, 3, 4] # Asegúrate de que coincidan con IDs de rutas válidos en tu BD
    usuarios = [1, 2, 3] # Asegúrate de que coincidan con IDs de usuarios válidos en tu BD
    
    lista_nuevos = []
    for i in range(cantidad):
        lista_nuevos.append({
            "id_viaje": siguiente_id + i,
            "fecha": "2026-11-15",
            "asientos_disponibles": 3,
            "costo": 20.0,
            "estado": estados[i % len(estados)],
            "id_ruta": rutas[i % len(rutas)],
            "id_usuario": usuarios[i % len(usuarios)]
        })
    
    db["viaje"].insert_many(lista_nuevos)
    print(f"¡Se insertaron {cantidad} viajes con ID secuencial para la tabla web!")

if __name__ == "__main__":
    poblar_viajes_con_id(50)