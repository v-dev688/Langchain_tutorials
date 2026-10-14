import os
from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS

load_dotenv()

# Use Google's current supported embedding model
embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")

texts = [
    "Quinton de Kock is regarded as one of modern cricket’s most destructive left-handed opening batsmen and a premier wicketkeeper.",
    "Quinton de Kock has represented and captained South Africa across all three international formats.",
    "At just 20 years old, Quinton de Kock became only the third player in ODI history—and the youngest ever—to hit three consecutive hundreds in a bilateral series, scoring 135, 106, and 101 against India.",
    "Behind the stumps, Quinton de Kock broke Adam Gilchrist’s record by becoming the fastest wicketkeeper in Test history to reach both 150 (35 matches) and 200 dismissals (47 matches)."
]

# Create FAISS index
vector_store = FAISS.from_texts(texts, embedding=embeddings)

# Similarity search : from a  list of texts it will give which are the most similer ones
results = vector_store.similarity_search("South Africa's all time wicket keeper", k=2)

for i, doc in enumerate(results, 1):
   
    print(f"{i}. {doc.page_content}\n")


retriever =vector_store.as_retreiver(search_kwargs={k:3})

retriever_tool=create_retriever_tool(retriever, name='kb_search', description="third player in ODI history—and the youngest ever—to hit three consecutive hundreds")
model = init_chat_model(
    model='google_genai:gemini-3.8-flash',
    temperature=0.1
)

checkpointer = InMemorySaver()

agent = create_agent(
    model=model,
    tools=[retriever_tool],
    system_prompt="You are a helpful weather assistant who always cracks jokes and is humorous while remaining helpful."
)
