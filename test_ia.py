import ollama

response = ollama.chat(
    model="llama3.2",
    messages=[
        {
            "role": "user",
            "content": "Saluda en una frase corta como el asistente de Velox UTVT.",
        },
    ],
)

print("Respuesta de la IA:", response["message"]["content"])
