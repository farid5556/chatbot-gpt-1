import openai
import os
from dotenv import load_dotenv

# Load API Key dari file .env
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

# Kirim prompt ke GPT
response = openai.ChatCompletion.create(
    model="gpt-3.5-turbo",  # Gunakan model gratisan dulu
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Jelaskan konsep supervised learning dengan sederhana."}
    ]
)

# Tampilkan hasilnya
print(response['choices'][0]['message']['content'])
