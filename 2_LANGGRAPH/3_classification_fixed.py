# this .py file is only for checking the routing



import uuid
from typing import Annotated, Literal
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field
from typing_extensions import TypedDict

load_dotenv()

# Initialize Gemini using langchain's init_chat_model
# Change this:
# llm = init_chat_model("google_genai:gemini-2.5-flash")

# To this:
llm = init_chat_model("google_genai:gemini-3.8-flash")


class IntentClassifier(BaseModel):
    message_intent: Literal["chat", "knowledge", "code"] = Field(
        ...,
        description="Classify whether the user wants to just chat, ask for knowledge, or change code in the project",
    )


class State(TypedDict):
    messages: Annotated[list, add_messages]
    message_intent: str | None


def classify_intent(state: State):
    structured_llm = llm.with_structured_output(IntentClassifier)
    result = structured_llm.invoke(
        [
            {
                "role": "system",
                "content": "Determine whether the user wants to chat ('chat'), retrieve knowledge ('knowledge'), or modify code ('code').",
            },
            {"role": "user", "content": state["messages"][-1].content},
        ]
    )
    return {"message_intent": result.message_intent}


def prompt_llm_chat(state: State):
    messages = [
        {"role": "system", "content": "You are a talkative, friendly chatbot for fun."}
    ] + state["messages"]
    response = llm.invoke(messages)
    return {"messages": [{"role": "assistant", "content": response.content}]}


def prompt_llm_rag(state: State):
    messages = [
        {"role": "system", "content": "No matter what the user says, say 'I am a RAG agent.'"}
    ] + state["messages"]
    response = llm.invoke(messages)
    return {"messages": [{"role": "assistant", "content": response.content}]}


def prompt_llm_code(state: State):
    messages = [
        {"role": "system", "content": "No matter what the user says, say 'I am a CODING agent.'"}
    ] + state["messages"]
    response = llm.invoke(messages)
    return {"messages": [{"role": "assistant", "content": response.content}]}


# Graph definition
graph_builder = StateGraph(State)
graph_builder.add_node("classifier", classify_intent)
graph_builder.add_node("chat_agent", prompt_llm_chat)
graph_builder.add_node("rag_agent", prompt_llm_rag)
graph_builder.add_node("coding_agent", prompt_llm_code)

graph_builder.add_edge(START, "classifier")
graph_builder.add_conditional_edges(
    "classifier",
    lambda state: state["message_intent"],
    {
        "chat": "chat_agent",
        "knowledge": "rag_agent",
        "code": "coding_agent",
    },
)

graph_builder.add_edge("chat_agent", END)
graph_builder.add_edge("rag_agent", END)
graph_builder.add_edge("coding_agent", END)

checkpointer = InMemorySaver()
graph = graph_builder.compile(checkpointer=checkpointer)

config = {"configurable": {"thread_id": str(uuid.uuid4())}}

while True:
    try:
        user_message = input("\nEnter message (or 'exit' to quit): ").strip()
        if user_message.lower() in ("exit", "quit"):
            break
        if not user_message:
            continue

        result = graph.invoke(
            {"messages": [{"role": "user", "content": user_message}]},
            config=config,
        )
        print(f"\nResponse: {result['messages'][-1].content}")
    except (KeyboardInterrupt, EOFError):
        break