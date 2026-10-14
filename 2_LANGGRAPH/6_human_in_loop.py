import os
import subprocess
import uuid
from typing import Annotated, Literal, TypedDict
from dotenv import load_dotenv

from langchain_core.documents import Document
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.types import Command, interrupt
from pydantic import BaseModel, Field

load_dotenv()

# 1. Initialize Groq Chat Model
# Groq models support structured output (.with_structured_output) natively
llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0.2,
    max_retries=3,
)

# 2. Local embeddings (no OpenAI key required, runs locally via HuggingFace)
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

KNOWLEDGE = [
    "Quinton de Kock opens with KL Rahul for Lucknow Super Giants, sharing an iconic 210-run unbeaten opening stand",
    "KL Rahul bats alongside Virat Kohli across multiple years for the Indian national cricket team",
    "Virat Kohli shares legendary batting partnerships with AB de Villiers for Royal Challengers Bengaluru.",
    "AB de Villiers spearheaded South Africa’s golden pace-and-power era with Dale Steyn",
    "Dale Steyn opened the fast-bowling attack alongside Kagiso Rabada, completing the generational South African cricket loop.",
]

vector_store = InMemoryVectorStore(embeddings)
vector_store.add_documents([Document(page_content=text) for text in KNOWLEDGE])


# 3. State & Intent Schemas
class IntentClassifier(BaseModel):
    message_intent: Literal["chat", "knowledge", "code"] = Field(
        ...,
        description="Classify whether the user wants to chat, ask for knowledge, or change code in the project",
    )


class State(TypedDict):
    messages: Annotated[list, add_messages]
    message_intent: str | None
    next_node: str | None


# 4. Graph Nodes
def classify_intent(state: State):
    structured_llm = llm.with_structured_output(IntentClassifier)
    result = structured_llm.invoke([
        {
            "role": "system",
            "content": (
                "You are an intent classification agent. Classify whether the user wants "
                "casual conversation ('chat'), to retrieve cricket knowledge ('knowledge'), "
                "or to run/edit code ('code')."
            ),
        },
        {"role": "user", "content": state["messages"][-1].content},
    ])
    return {"message_intent": result.message_intent}


def accept_coding(state: State):
    user_prompt = state["messages"][-1].content
    decision = interrupt(
        f"About to run Claude Code with request:\n\n{user_prompt}\n\nApprove? (yes/no, or type a revised request)"
    )

    text = str(decision).strip().lower()
    if text in ["y", "yes", "approve", "ok"]:
        return {}

    if text in ["n", "no", "deny", "cancel"]:
        return {
            "messages": [{"role": "assistant", "content": "Coding request was denied by the user."}],
            "next_node": "denied",
        }

    # Pass the revised prompt as the next user message
    return {"messages": [{"role": "user", "content": text}]}


def prompt_llm_chat(state: State):
    messages = [{"role": "system", "content": "You are a friendly, talkative chatbot."}] + state["messages"]
    response = llm.invoke(messages)
    return {"messages": [{"role": "assistant", "content": response.content}]}


def prompt_llm_rag(state: State):
    query = state["messages"][-1].content
    documents = vector_store.similarity_search(query, k=3)
    context = "\n".join(f"- {doc.page_content}" for doc in documents)

    system_prompt = (
        "You are a cricket knowledge assistant. Answer the user query using only the following context:\n"
        f"{context}"
    )
    messages = [{"role": "system", "content": system_prompt}] + state["messages"]
    response = llm.invoke(messages)
    return {"messages": [{"role": "assistant", "content": response.content}]}


def prompt_llm_code(state: State):
    user_prompt = state["messages"][-1].content
    workspace = os.path.join(os.path.dirname(os.path.abspath(__file__)), "workspace")
    os.makedirs(workspace, exist_ok=True)

    result = subprocess.run(
        ["claude", "-p", user_prompt, "--permission-mode", "acceptEdits"],
        cwd=workspace,
        capture_output=True,
        text=True,
        shell=True,
    )
    output = result.stdout.strip() or result.stderr.strip() or "Command completed with no output."
    return {"messages": [{"role": "assistant", "content": output}]}


# 5. Graph Assembly
graph_builder = StateGraph(State)

graph_builder.add_node("classifier", classify_intent)
graph_builder.add_node("chat_agent", prompt_llm_chat)
graph_builder.add_node("rag_agent", prompt_llm_rag)
graph_builder.add_node("accept_coding", accept_coding)
graph_builder.add_node("coding_agent", prompt_llm_code)

graph_builder.add_edge(START, "classifier")

# Route 'code' through the approval gate before executing
graph_builder.add_conditional_edges(
    "classifier",
    lambda state: state["message_intent"],
    {"chat": "chat_agent", "knowledge": "rag_agent", "code": "accept_coding"},
)

# Route to coding_agent if approved, or END if denied
graph_builder.add_conditional_edges(
    "accept_coding",
    lambda state: "end" if state.get("next_node") == "denied" else "coding_agent",
    {"end": END, "coding_agent": "coding_agent"},
)

graph_builder.add_edge("chat_agent", END)
graph_builder.add_edge("rag_agent", END)
graph_builder.add_edge("coding_agent", END)

checkpointer = InMemorySaver()
graph = graph_builder.compile(checkpointer=checkpointer)

# 6. Interactive CLI Loop
config = {"configurable": {"thread_id": str(uuid.uuid4())}}

while True:
    try:
        user_message = input("\nEnter message: ").strip()
        if not user_message:
            continue
        if user_message.lower() in ["exit", "quit"]:
            break

        result = graph.invoke({"messages": [{"role": "user", "content": user_message}]}, config=config)

        while "__interrupt__" in result:
            prompt = result["__interrupt__"][0].value
            decision = input(f"\n[APPROVAL PROMPT]\n{prompt}\n> ")
            result = graph.invoke(Command(resume=decision), config=config)

        print(f"\nAssistant: {result['messages'][-1].content}")

    except (KeyboardInterrupt, EOFError):
        print("\nExiting.")
        break