import time
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

from langchain.agents import create_agent
from langchain.agents.middleware import AgentMiddleware, AgentState
from langchain_core.messages import SystemMessage, HumanMessage

class HooksDemo(AgentMiddleware):
    def __init__(self):
        super().__init__()
        self.start_time = 0.0

    def before_agent(self, state: AgentState, runtime):
        self.start_time = time.time()
        print("--- [Hook] before_agent ---")

    def before_model(self, state: AgentState, runtime):
        print("--- [Hook] before_model ---")

    def after_model(self, state: AgentState, runtime):
        print("--- [Hook] after_model ---")

    def after_agent(self, state: AgentState, runtime):
        elapsed = time.time() - self.start_time
        print(f"--- [Hook] after_agent (Total time: {elapsed:.2f}s) ---")


agent = create_agent(
    model='google_genai:gemini-3.8-flash',
    middleware=[HooksDemo()]
)

response = agent.invoke({
    'messages': [
        SystemMessage(content="You are a helpful assistant."),
        HumanMessage(content="What is PCA in 2 lines?")
    ]
})

print("\nResponse:")
print(response['messages'][-1].content)