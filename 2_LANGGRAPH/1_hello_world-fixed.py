from dotenv import load_dotenv
from langchain_core.messages import SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import END, START, MessagesState, StateGraph

load_dotenv()

llm = ChatGoogleGenerativeAI(model="gemini-3.8-flash")


# Node 1: Inserts the system instruction into the state
def add_prompt(state: MessagesState):
  system_instruction = SystemMessage(
      content="You are a helpful cricket coach. Always reply with enthusiasm."
  )
  return {"messages": [system_instruction]}


# Node 2: Takes all messages accumulated in state and calls the LLM
def prompt_llm(state: MessagesState):
  response = llm.invoke(state["messages"])
  return {"messages": [response]}


# Build the Graph
graph_builder = StateGraph(MessagesState)

# 1. Register both nodes
graph_builder.add_node("add_prompt", add_prompt)
graph_builder.add_node("prompt_llm", prompt_llm)

# 2. Wire the flow: START -> add_prompt -> prompt_llm -> END
graph_builder.add_edge(START, "add_prompt")
graph_builder.add_edge("add_prompt", "prompt_llm")
graph_builder.add_edge("prompt_llm", END)

graph = graph_builder.compile()

user_message = input("Enter message: ")
result = graph.invoke(
    {"messages": [{"role": "user", "content": user_message}]}
)

print("\nAssistant Response:")
print(result["messages"][-1].content)