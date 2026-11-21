import os
from urllib.parse import quote_plus
from dotenv import load_dotenv
from pymongo import MongoClient

# Carga las variables desde el archivo .env en la raíz del proyecto
load_dotenv()

MONGO_USER = os.getenv("MONGO_USER", "admin")
MONGO_PASSWORD = os.getenv("MONGO_PASSWORD", "mellamo99")
MONGO_CLUSTER = os.getenv("MONGO_CLUSTER", "cluster0.scrszhd.mongodb.net")
MONGO_DB_NAME = os.getenv("MONGO_DB_NAME", "velox_utvt")
MONGO_COLLECTION = os.getenv("MONGO_COLLECTION", "capibara")

usuario_seguro = quote_plus(MONGO_USER)
password_seguro = quote_plus(MONGO_PASSWORD)

MONGO_URI = (
    f"mongodb+srv://{usuario_seguro}:{password_seguro}"
    f"@{MONGO_CLUSTER}/{MONGO_DB_NAME}?retryWrites=true&w=majority&appName=Cluster0"
)

client = MongoClient(MONGO_URI)
db = client[MONGO_DB_NAME]
collection = db[MONGO_COLLECTION]