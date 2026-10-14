import os
import requests
from dotenv import load_dotenv

from langchain.chat_models import init_chat_model
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

load_dotenv()

model = init_chat_model(
    model='google_genai:gemini-3.8-flash',
    temperature=0.1
)

conversation = [
    SystemMessage(content="You are a helpful and knowledgeable cricket assistant."),
    AIMessage(content="Hi Vasudev, how are you today?"),
    HumanMessage(content="Why did Quinton de Kock and Temba Bavuma argue last week?"),
    AIMessage(content="It was regarding a DRS review decision during the middle overs."),
    HumanMessage(content="Can you elaborate more?")
]

response = model.invoke(conversation)

print(response.content)