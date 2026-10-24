# Importa el módulo 'os' para interactuar con el sistema operativo y leer variables de entorno.
import os

# Importa 'Path' de la librería 'pathlib' para manejar rutas de archivos y directorios de forma independiente del sistema operativo.
from pathlib import Path

# Importa 'quote_plus' para codificar caracteres especiales (como @, #, etc.) en contraseñas para URLs.
from urllib.parse import quote_plus

# Importa 'load_dotenv' de la librería 'python-dotenv' para cargar variables desde un archivo .env.
from dotenv import load_dotenv

# Importa 'MongoClient' de PyMongo para conectarse y gestionar bases de datos MongoDB.
from pymongo import MongoClient


# Define la ruta absoluta del archivo .env buscando un nivel arriba de donde está este script actual.
env_path = Path(__file__).resolve().parent.parent / ".env"

# Carga las variables de entorno definidas en el archivo .env encontrado en la ruta anterior.
load_dotenv(dotenv_path=env_path)

# Obtiene el nombre de usuario de MongoDB desde las variables de entorno.
user = os.getenv("MONGO_USER")

# Obtiene la contraseña de MongoDB y aplica 'quote_plus' para evitar errores por caracteres especiales; si está vacía, usa una cadena vacía.
password = quote_plus(os.getenv("MONGO_PASSWORD") or "")

# Obtiene la dirección del servidor o clúster de MongoDB Atlas desde las variables de entorno.
cluster = os.getenv("MONGO_CLUSTER")

# Obtiene el nombre de la base de datos que se utilizará en el proyecto.
database_name = os.getenv("MONGO_DB_NAME")

# Valida que ninguna de las credenciales esenciales (usuario, contraseña, clúster y base de datos) esté vacía o ausente.
if not all([user, password, cluster, database_name]):
  # Si falta alguna variable, lanza un error crítico deteniendo la ejecución.
  raise ValueError("Faltan variables necesarias en el archivo .env")

# Construye la cadena de conexión oficial (URI) utilizando formato estándar para MongoDB Atlas con autenticación.
mongo_uri = f"mongodb+srv://{user}:{password}@{cluster}/"

# Bloque protegido para intentar la conexión con la base de datos y manejar posibles errores de red o credenciales.
try:
  # Inicializa el cliente de MongoDB configurando un tiempo límite de espera de 5 segundos (5000 ms).
  client = MongoClient(mongo_uri, serverSelectionTimeoutMS=5000)

  # Ejecuta un comando de prueba ('ping') al servidor para verificar que la conexión se estableció correctamente.
  client.admin.command("ping")

  # Selecciona y almacena la base de datos específica para que esté lista para usarse en el proyecto.
  db = client[database_name]

  # Imprime un mensaje en consola confirmando el éxito de la conexión junto con el nombre de la base de datos.
  print(f"Conexión exitosa a MongoDB - Base de datos: '{database_name}'")

# Si ocurre cualquier excepción (error de red, contraseña incorrecta, etc.), se captura aquí.
except Exception as error:
  # Imprime el mensaje de error detallado en la consola.
  print(f"Error al conectar con MongoDB: {error}")

  # Detiene por completo la ejecución del programa pasando el error obtenido.
  raise SystemExit(error)