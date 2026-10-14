from dataclasses import dataclass
from dotenv import load_dotenv

load_dotenv()

from langchain.agents import create_agent
from langchain.agents.middleware import ModelRequest, dynamic_prompt

@dataclass
class Context:
    user_role: str

@dynamic_prompt
def user_role_prompt(request: ModelRequest) -> str:
    user_role = request.runtime.context.user_role
    base_prompt = "You are a helpful and very concise assistant."

    match user_role:
        case 'expert':
            return f"{base_prompt} Provide a detailed technical response."
        case 'beginner':
            return f"{base_prompt} Keep your explanations simple and basic."
        case 'child':
            return f"{base_prompt} Keep your explanations simple and basic like you are explaining to a 5-year-old child."
        case _:
            return base_prompt

# Use Gemini here instead of OpenAI
agent = create_agent(
    model='google_genai:gemini-3.8-flash',
    middleware=[user_role_prompt],
    context_schema=Context
)

response = agent.invoke(
    {
        'messages': [{'role': 'user', 'content': 'Explain PCA'}]
    },
    context=Context(user_role='expert')
)

print(response['messages'][-1].content)