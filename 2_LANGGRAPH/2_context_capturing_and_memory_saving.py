# this example is about how to add a simple prompt in between
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langgraph.graph import MessagesState, StateGraph, START, END
from langgraph.checkpoint.memory import InMemorySaver
import uuid



load_dotenv()

llm = init_chat_model('openai:gpt-4.1-mini')


def prompt_llm(state: MessagesState):    # this takes current state of the graph, we are passing state from node to node
                                         # by default state is list of messages, and messages are appeneded
    response= llm.invoke(state['messages'])
    return {'messages':[response]}

graph_builder = StateGraph(MessagesState) # this is for builing the graph, 

graph_builder.add_node(prompt_llm) # start node and end node are added by default
graph_builder.add_edge(START,"prompt_llm")
graph_builder.add_edge("prompt_llm",END)

checkpointer= InMemorySaver()
graph=graph_builder.compile()

config ={'configurable': {'thread-id':uuid.uuid4()}}

while True:
    user_message = input("Enter message")
    graph.invoke({'messages':[{'role':'user','content':user_message}]}, config=config)