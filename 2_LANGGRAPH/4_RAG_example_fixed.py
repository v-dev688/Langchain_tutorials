import uuid
from typing import Annotated, Literal, TypedDict
from dotenv import load_dotenv
from pydantic import BaseModel, Field

from langchain.chat_models import init_chat_model
from langchain_core.documents import Document
from langchain_community.vectorstores import SKLearnVectorStore
from langchain_core.embeddings import Embeddings
from sklearn.feature_extraction.text import TfidfVectorizer

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages

load_dotenv()

# Fast, free LLM provider with no restrictive 20-call free-tier block
llm = init_chat_model(
    "groq:openai/gpt-oss-20b",
    max_retries=5,
)
KNOWLEDGE = [
    "Quinton de Kock opens with KL Rahul for Lucknow Super Giants, sharing an iconic 210-run unbeaten opening stand",
    "KL Rahul bats alongside Virat Kohli across multiple years for the Indian national cricket team",
    "Virat Kohli shares legendary batting partnerships with AB de Villiers for Royal Challengers Bengaluru.",
    "AB de Villiers spearheaded South Africa's golden pace-and-power era with Dale Steyn",
    "Dale Steyn opened the fast-bowling attack alongside Kagiso Rabada, completing the generational South African cricket loop.",
]

# TF-IDF Embeddings (no GPU, Torch, or external embeddings API required)
class LocalTfidfEmbeddings(Embeddings):
    def __init__(self, corpus: list[str]):
        self.vectorizer = TfidfVectorizer().fit(corpus)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self.vectorizer.transform(texts).toarray().tolist()

    def embed_query(self, text: str) -> list[float]:
        return self.vectorizer.transform([text]).toarray()[0].tolist()

embeddings = LocalTfidfEmbeddings(KNOWLEDGE)
docs = [Document(page_content=text) for text in KNOWLEDGE]
vector_store = SKLearnVectorStore.from_documents(docs, embeddings)


class IntentClassifier(BaseModel):
    message_intent: Literal["chat", "knowledge", "code"] = Field(
        ...,
        description="Classify whether the user wants to chat, ask for knowledge, or change code.",
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
                "content": 'Classify whether the user wants general chat ("chat"), domain knowledge/retrieval ("knowledge"), or programming assistance ("code").',
            },
            {"role": "user", "content": state["messages"][-1].content},
        ]
    )
    return {"message_intent": result.message_intent}


def prompt_llm_chat(state: State):
    messages = [
        {"role": "system", "content": "You are a conversational chatbot. Prefix your answer with '[Chat Agent]: '"}
    ] + state["messages"]
    response = llm.invoke(messages)
    return {"messages": [{"role": "assistant", "content": response.content}]}


def prompt_llm_rag(state: State):
    query = state["messages"][-1].content
    documents = vector_store.similarity_search(query, k=2)
    context = "\n".join(f"- {doc.page_content}" for doc in documents)

    messages = [
        {
            "role": "system",
            "content": f"Answer based on this retrieved context:\n{context}\n\nPrefix your answer with '[RAG Agent]: '",
        }
    ] + state["messages"]
    response = llm.invoke(messages)
    return {"messages": [{"role": "assistant", "content": response.content}]}


def prompt_llm_code(state: State):
    messages = [
        {
            "role": "system",
            "content": "You are a coding assistant. Prefix your answer with '[Code Agent]: '",
        }
    ] + state["messages"]
    response = llm.invoke(messages)
    return {"messages": [{"role": "assistant", "content": response.content}]}


graph_builder = StateGraph(State)
graph_builder.add_node("classifier", classify_intent)
graph_builder.add_node("chat_agent", prompt_llm_chat)
graph_builder.add_node("rag_agent", prompt_llm_rag)
graph_builder.add_node("coding_agent", prompt_llm_code)

graph_builder.add_edge(START, "classifier")
graph_builder.add_conditional_edges(
    "classifier",
    lambda state: state["message_intent"],
    {"chat": "chat_agent", "knowledge": "rag_agent", "code": "coding_agent"},
)

graph_builder.add_edge("chat_agent", END)
graph_builder.add_edge("rag_agent", END)
graph_builder.add_edge("coding_agent", END)

checkpointer = InMemorySaver()
graph = graph_builder.compile(checkpointer=checkpointer)

config = {"configurable": {"thread_id": str(uuid.uuid4())}}

print("--- Agent Graph Ready (Type 'exit' to quit) ---")
while True:
    user_message = input("\nEnter message: ")
    if user_message.strip().lower() in {"exit", "quit"}:
        break

    result = graph.invoke(
        {"messages": [{"role": "user", "content": user_message}]},
        config=config,
    )
    print(result["messages"][-1].content)