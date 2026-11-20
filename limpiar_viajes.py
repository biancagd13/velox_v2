from database import db

# 1. Contamos cuántos viajes hay antes de limpiar
total_actual = db["viaje"].count_documents({})
print(f"Total de viajes actuales en MongoDB: {total_actual}")

# 2. Si quieres vaciar la tabla por completo:
# db["viaje"].delete_many({})
# print("¡Colección vaciada por completo!")

# 3. O si prefieres conservar únicamente los últimos 50 registros insertados:
# Obtenemos los IDs de los 50 documentos más recientes ordenados por _id descendente
ultimos_50 = list(db["viaje"].find({}, {"_id": 1}).sort("_id", -1).limit(50))

if ultimos_50:
    # Extraemos los _id que queremos conservar
    ids_a_conservar = [doc["_id"] for doc in ultimos_50]
    
    # Borramos todos los documentos cuyo _id NO esté en la lista de los 50 más recientes
    resultado = db["viaje"].delete_many({"_id": {"$nin": ids_a_conservar}})
    print(f"¡Limpieza exitosa! Se eliminaron {resultado.deleted_count} registros antiguos.")
    print(f"Han quedado exactamente {db['viaje'].count_documents({})} registros en la base de datos.")
else:
    print("La colección ya está vacía.")