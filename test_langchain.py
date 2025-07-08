import os
import dotenv
from pathlib import Path

from langchain_core.messages import AIMessage, HumanMessage
from langchain_community.document_loaders.text import TextLoader
from langchain_community.document_loaders import (
    WebBaseLoader, 
    PyPDFLoader, 
    Docx2txtLoader,
)
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
#from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

dotenv.load_dotenv()
import os

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Load docs

doc_paths = [
    "docs/test_rag.pdf",
    "docs/test_rag.docx",
]

docs = [] 
for doc_file in doc_paths:
    file_path = Path(doc_file)

    try:
        if doc_file.endswith(".pdf"):
            loader = PyPDFLoader(file_path)
        elif doc_file.endswith(".docx"):
            loader = Docx2txtLoader(file_path)
        elif doc_file.endswith(".txt") or doc_file.name.endswith(".md"):
            loader = TextLoader(file_path)
        else:
            print(f"Document type {doc_file.type} not supported.")
            continue

        docs.extend(loader.load())

    except Exception as e:
        print(f"Error loading document {doc_file.name}: {e}")


# Load URLs

url = "https://docs.streamlit.io/develop/quick-reference/release-notes"
try:
    loader = WebBaseLoader(url)
    docs.extend(loader.load())

except Exception as e:
    print(f"Error loading document from {url}: {e}")


# Split docs

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=5000,
    chunk_overlap=1000,
)

document_chunks = text_splitter.split_documents(docs)

# Tokenize and load the documents to the vector store

vector_db = Chroma.from_documents(
    documents=document_chunks,
    embedding=OpenAIEmbeddings(),
)

# Retrieve

def _get_context_retriever_chain(vector_db, llm):
    retriever = vector_db.as_retriever()
    prompt = ChatPromptTemplate.from_messages([
        MessagesPlaceholder(variable_name="messages"),
        ("user", "{input}"),
        ("user", "Given the above conversation, generate a search query to look up in order to get inforamtion relevant to the conversation, focusing on the most recent messages."),
    ])
    retriever_chain = create_history_aware_retriever(llm, retriever, prompt)

    return retriever_chain


def get_conversational_rag_chain(llm):
    retriever_chain = _get_context_retriever_chain(vector_db, llm)

    prompt = ChatPromptTemplate.from_messages([
        ("system",
        """You are a helpful assistant. You will have to answer to user's queries.
        You will have some context to help with your answers, but now always would be completely related or helpful.
        You can also use your knowledge to assist answering the user's queries.\n
        {context}"""),
        MessagesPlaceholder(variable_name="messages"),
        ("user", "{input}"),
    ])
    stuff_documents_chain = create_stuff_documents_chain(llm, prompt)

    return create_retrieval_chain(retriever_chain, stuff_documents_chain)


llm_stream_openai = ChatOpenAI(
    model="gpt-4o",  # Here you could use "o1-preview" or "o1-mini" if you already have access to them
    temperature=0.3,
    streaming=True,
)

llm_stream = llm_stream_openai  # Select between OpenAI and Anthropic models for the response

messages = [
    {"role": "user", "content": "Hi"},
    {"role": "assistant", "content": "Hi there! How can I assist you today?"},
    {"role": "user", "content": "What is the latest version of Streamlit?"},
]
messages = [HumanMessage(content=m["content"]) if m["role"] == "user" else AIMessage(content=m["content"]) for m in messages]

conversation_rag_chain = get_conversational_rag_chain(llm_stream)
response_message = "*(RAG Response)*\n"
for chunk in conversation_rag_chain.pick("answer").stream({"messages": messages[:-1], "input": messages[-1].content}):
    response_message += chunk
    print(chunk, end="", flush=True)

messages.append({"role": "assistant", "content": response_message})