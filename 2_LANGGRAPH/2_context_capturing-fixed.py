import uuid
from dotenv import load_dotenv
from langchain_core.messages import SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, MessagesState, StateGraph

load_dotenv()

llm = ChatGoogleGenerativeAI(model="gemini-3.8-flash")


def prompt_llm(state: MessagesState):
  # Prepend your system instructions to the ongoing message history
  system_prompt = SystemMessage(
      content="You are a helpful, concise AI assistant."
  )
  messages_to_send = [system_prompt] + state["messages"]

  response = llm.invoke(messages_to_send)
  return {"messages": [response]}


graph_builder = StateGraph(MessagesState)

# Register node and edges
graph_builder.add_node("prompt_llm", prompt_llm)
graph_builder.add_edge(START, "prompt_llm")
graph_builder.add_edge("prompt_llm", END)

# Attach checkpointer for persistent memory across loop iterations
checkpointer = InMemorySaver()
graph = graph_builder.compile(checkpointer=checkpointer)

# LangGraph config requires 'thread_id' (with underscore)
config = {"configurable": {"thread_id": str(uuid.uuid4())}}

print("Chatbot initialized. Type 'exit' or 'quit' to stop.\n")

while True:
  user_message = input("\nEnter message: ")
  if user_message.lower().strip() in ["exit", "quit"]:
    break

  # Run the graph and print the latest response
  result = graph.invoke(
      {"messages": [{"role": "user", "content": user_message}]}, config=config
  )

  print("\nAI:", result["messages"][-1].content)