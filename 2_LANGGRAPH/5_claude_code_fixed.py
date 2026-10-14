# this example is about how to add a simple prompt in between
# this .py file is only for checking the routing
import os
import subprocess

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from typing import TypedDict, Annotated, Literal
from langchain_openai import Opena
from langchain_core.vectorstores import InMemoryVectorStore

from langchain_openai import OpenAIEmbeddings
from pydantic import BaseModel, Field
from langgraph.graph import  MessagesState,StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import InMemorySaver
import uuid



load_dotenv()

llm = init_chat_model(
    "google_genai:gemini-2.0-flash", # or gemini-1.5-flash / gemini-3.8-flash
    max_retries=5,
)

KNOWLEDGE=[
    "Quinton de Kock opens with KL Rahul for Lucknow Super Giants, sharing an iconic 210-run unbeaten opening stand",
    "KL Rahul bats alongside Virat Kohli across multiple years for the Indian national cricket team",
    "Virat Kohli shares legendary batting partnerships with AB de Villiers for Royal Challengers Bengaluru.",
    "AB de Villiers spearheaded South Africa’s golden pace-and-power era with Dale Steyn",
    "Dale Steyn opened the fast-bowling attack alongside Kagiso Rabada, completing the generational South African cricket loop."
]


vector_store =InMemoryVectorStore(OpenAIEmbeddings('text-embedding-3-small'))
vector_store.add_documents([Document(page_content=text) for text in KNOWLEDGE])


class IntentClassifier(BaseModel):
    message_intent:Literal['chat','knowledge','code']=Field(..., description="Classify whther the user want to just chat, ask for knowoledge or change code in project")

class State(TypedDict):
    messages= Annotated[list, add_messages]
    message_intent= str | None

def classify_intent(state:State):
    structured_llm= llm.with_structured_output(IntentClassifier)
    result= structured_llm.invoke([
        {'role':'system','content':'Determine/ classify whether a user want to chat("chat"), retrieve a knowledge ("knowledge") or change code("code").'},
        {'role':'user','content':state.messages[-1].content}
    ])

    return {'message_intent':result.message_intent}

def prompt_llm_chat(state:State):
    messages=[{'role':'system','content':'You are a talkative chatbot fo fun. be nice'}] + state['messages']
    response = llm.invoke(messages)

    return {'messages':[{'role':'assistant','content':response.content}]}

def prompt_llm_rag(state:State):
    query= state['messages'[-1]].content
    documents= vector_store.similarity_search(query, k=3)
    context ="\n".join(f'-{doc.page_content}' for doc in documents)

    messages=[{'role':'system','content':'No matter what user says say i am RAG agent'}] + state['messages']
    response = llm.invoke(messages)

    return {'messages':[{'role':'assistant','content':response.content}]}

def prompt_llm_code(state:State):
    user_prompt = state["messages"][-1].content
    workspace= os.path(os.path.dirname(os.path.abspath(__file__)), 'workspace')

    result= subprocess.run(
        ['claude','-p',user_prompt,'--permissionMode','acceptEdits']
        cwd=workspace
        capture_output=True,
        text=True

    )
    output= result.stdout.strip() or result.stderr.strip()


    messages=[{'role':'system','content':'No matter what user says say i am CODING agent'}] + state['messages']
    response = llm.invoke(messages)

    return {'messages':[{'role':'assistant','content':output}]}



graph_builder = StateGraph(State)
graph_builder.add_node('classifier',classify_intent)
graph_builder.add_node('chat_agent',prompt_llm_chat)
graph_builder.add_node('rag_agent',prompt_llm_rag)
graph_builder.add_node('coding_agent', prompt_llm_code)

graph_builder.add_edge(START,'classifier')
graph_builder.add_conditional_edges('classifier',lambda state:state['message_intent'], {'chat':'chat_agent','knowledge':'rag_agent','code':'coding_agent'})

graph_builder.add_edge('chat_agent', END)
graph_builder.add_edge('rag_agent',END)
graph_builder.add_edge('coding_agent',END)

checkpointer = InMemorySaver()
graph = graph_builder.compile(checkpointer=checkpointer)

config={'configurable':{'thread_id': uuid.uuid4()}}

while True:
    user_message =input("Enter message :")
    result = graph.invoke({'messages':[{'role':'user','content':user_message}]}, config= config)

    print(result['messages'][-1].content)
