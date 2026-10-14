from dataclasses import dataclass
import requests
from dotenv import load_dotenv
from pydantic import BaseModel, Field

from langchain.agents import create_agent
from langchain.tools import tool, ToolRuntime
from langchain.chat_models import init_chat_model
from langgraph.checkpoint.memory import InMemorySaver

load_dotenv()

@dataclass
class Context:
    user_id: str

# Pydantic model ensures proper JSON schema generation for the LLM
class ResponseFormat(BaseModel):
    summary: str = Field(description="A brief summary of the weather")
    temperature_celsius: float = Field(description="Temperature in Celsius")
    temperature_fahrenheit: float = Field(description="Temperature in Fahrenheit")
    humidity: float = Field(description="Humidity percentage")

@tool('get_weather', description="Return weather information for a given city", return_direct=False)
def get_weather(city: str):
    response = requests.get(f'https://wttr.in/{city}?format=j1')
    return response.json()

@tool('locate_user', description="Look up a user's city based on runtime context")
def locate_user(runtime: ToolRuntime[Context]):
    match runtime.context.user_id:
        case 'ABC123':
            return 'vienna'
        case 'XYZ456':
            return 'london'
        case 'HJKL111':
            return 'paris'
        case _:
            return 'Unknown'

model = init_chat_model(
    model='google_genai:gemini-3.8-flash',
    temperature=0.1
)

checkpointer = InMemorySaver()

agent = create_agent(
    model=model,
    tools=[get_weather, locate_user],
    system_prompt="You are a helpful weather assistant who always cracks jokes and is humorous while remaining helpful.",
    context_schema=Context,
    response_format=ResponseFormat,
    checkpointer=checkpointer
)

config = {"configurable": {"thread_id": "session_1"}}

response = agent.invoke(
    {
        "messages": [
            {"role": "user", "content": "What is weather like in Vienna?"}
        ]
    },
    config=config,
    context=Context(user_id='ABC123')
)

print(response)
print("\n--- Structured Response ---")
print(response['structured_response'])
print("Summary:", response['structured_response'].summary)
print("Temperature (C):", response['structured_response'].temperature_celsius)