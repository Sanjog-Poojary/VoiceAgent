import os
from dotenv import load_dotenv
load_dotenv()
from google import genai
from pydantic import BaseModel

class Response(BaseModel):
    a: str

client = genai.Client()
response = client.models.generate_content(
    model='gemini-flash-latest',
    contents='say hi',
    config=genai.types.GenerateContentConfig(
        response_mime_type='application/json',
        response_schema=Response,
    ),
)
print(response.text)
