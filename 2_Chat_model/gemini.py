import pytorch as torch

from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

load_dotenv()  # Load environment variables from .env file

chat = ChatGoogleGenerativeAI(
    model='gemini-2.5-flash',
)

response = chat.invoke('what is Machine Learning?')

print(response.content)