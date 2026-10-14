from dotenv import load_dotenv

load_dotenv()

from langchain.agents import create_agent
from langchain.agents.middleware import ModelRequest, ModelResponse, wrap_model_call
from langchain.chat_models import init_chat_model
from langchain_core.messages import SystemMessage, HumanMessage

# Initialize Gemini models
basic_model = init_chat_model(
    model='google_genai:gemini-3.8-flash',
    temperature=0.1
)
advanced_model = init_chat_model(
    model='google_genai:gemini-3.1-pro-preview',
    temperature=0.1
)
@wrap_model_call
def dynamic_model_selection(request: ModelRequest, handler) -> ModelResponse:
    message_count = len(request.state['messages'])
    
    # Route based on conversation turn length
    if message_count > 3:
        request.model = basic_model
    else:
        request.model = advanced_model

    return handler(request)

# Build agent with default base model and dynamic model-swapping middleware
agent = create_agent(
    model=basic_model,
    middleware=[dynamic_model_selection]
)

response = agent.invoke({
    'messages': [
        SystemMessage(content="You are a helpful assistant."),
        HumanMessage(content="what is 1+1")  # Comma was missing here
    ]
})

print("Response:", response['messages'][-1].content)
print("Model used:", response['messages'][-1].response_metadata.get('model_name'))