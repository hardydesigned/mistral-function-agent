import chromadb
import requests
from langchain.text_splitter import RecursiveCharacterTextSplitter
import ollama
import asyncio

client = chromadb.Client()
collection = client.create_collection(name="docs")

def split_document(path: str):
    with open(path, 'r', encoding="UTF-8") as file:
        text = file.read()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=250,
        chunk_overlap=40
    )
    output = splitter.create_documents([text])
    text_arr = [chunk.page_content for chunk in output]
    return text_arr


def create_embeddings(chunks):
    for i, d in enumerate(chunks):
        embeddings = ollama.embeddings(
            model='nomic-embed-text',
            prompt=d
        )
        collection.add(
    ids=[str(i)],
    embeddings=embeddings["embedding"],
    documents=[d]
  )
       
def create_embedding(chunk):
    embeddings = ollama.embeddings(
            model='nomic-embed-text',
            prompt=chunk
        )
       
    return embeddings['embedding']
    



def retrieve_matches(embedding):
    results = collection.query(
    query_embeddings=[embedding],
  n_results=5
)
    return results['documents']


def generate_chat_response(context, query):
    response = ollama.chat(
        model='llama3.1',
        messages=[
            {
                "role": "user",
                "content": f"Handbook context: {context} - Question: {query}"
            }
        ]
    )
    return response['message']['content']

def rag_agent(query):
    # 1. Getting the user input

    handbook_chunks = split_document('handbook.txt')
    data = create_embeddings(handbook_chunks)

    embedding = create_embedding(query)

    # 3. Retrieving similar embeddings / text chunks (aka "context")
    context = retrieve_matches(embedding)

    # 4. Combining the input and the context in a prompt and using the chat API to generate a response
    response = generate_chat_response(context, query)
    print(response)

