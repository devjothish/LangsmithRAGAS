from langchain_openai import ChatOpenAI
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Initialize the model with the correct model name
llm = ChatOpenAI(model="gpt-4")

# Test the model
response = llm.invoke("Hello, world!")
print(response.content)