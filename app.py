from flask import Flask, render_template, jsonify, request
from langchain_pinecone import PineconeVectorStore
from langchain_core.globals import set_llm_cache
from langchain_core.exceptions import OutputParserException
from langchain_openai import ChatOpenAI
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from src.helper import download_hugging_face_embeddings
from src.prompt import system_prompt
from dotenv import load_dotenv
from openai import AuthenticationError
from threading import Lock
import os


app = Flask(__name__)
load_dotenv()

# --- Startup API Key Validation ---
# Fail fast if essential keys are not configured.
PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY')
OPENROUTER_API_KEY = os.environ.get('OPENROUTER_API_KEY')
if not PINECONE_API_KEY or 'your' in PINECONE_API_KEY.lower():
    raise RuntimeError("PINECONE_API_KEY is missing or a placeholder. Please set it in your .env file.")
if not OPENROUTER_API_KEY or 'your' in OPENROUTER_API_KEY.lower():
    raise RuntimeError("OPENROUTER_API_KEY is missing or a placeholder. Please set it in your .env file.")


# Disable LangChain's in-memory cache for a stateless request-response model
set_llm_cache(None)

# Use a global variable to hold the chain, but initialize it lazily
rag_chain = None
_lock = Lock()

def create_rag_chain():
    """Creates the RAG chain for the chatbot."""
    embeddings = download_hugging_face_embeddings()

    index_name = os.environ.get('INDEX_NAME', 'medical-chatbot')
    # Initialize PineconeVectorStore for retrieval
    docsearch = PineconeVectorStore.from_existing_index(
        index_name=index_name,
        embedding=embeddings
    )

    retriever = docsearch.as_retriever(search_type="similarity", search_kwargs={"k": 3})

    chat_model = ChatOpenAI(
        model="openai/gpt-4o-mini",
        base_url="https://openrouter.ai/api/v1",
        api_key=OPENROUTER_API_KEY,
        temperature=0,
    )
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("human", "{input}"),
        ]
    )

    combine_docs_chain = create_stuff_documents_chain(chat_model, prompt)
    return create_retrieval_chain(retriever, combine_docs_chain)

def get_rag_chain():
    """
    Initializes and returns the RAG chain, creating it only once.
    """
    global rag_chain
    # First check without a lock for performance
    if rag_chain is None:
        with _lock:
            # Double-check inside the lock to prevent race conditions
            if rag_chain is None:
                rag_chain = create_rag_chain()
    return rag_chain

@app.route("/")
def index():
    return render_template('chat.html')

@app.route("/get", methods=["GET", "POST"])
def chat():
    msg = request.form.get("msg", "").strip()
    if not msg:
        return jsonify({"error": "Please enter a question."}), 400

    try:
        print(f"Received message: {msg}")
        chain = get_rag_chain()
        response = chain.invoke({"input": msg})
        answer = response.get("answer", "Sorry, I could not find an answer.")
        print(f"Response: {answer}")
        return jsonify({"answer": answer})
    except AuthenticationError:
        # This catches invalid API key errors from OpenRouter/OpenAI
        error_msg = "Authentication failed. Please check if your OPENROUTER_API_KEY is correct and valid."
        print(f"An error occurred: {error_msg}")
        return jsonify({"error": error_msg}), 500
    except OutputParserException:
        # This can happen if the LLM response is not in the expected format
        error_msg = "Sorry, I had trouble formatting the response. Please try rephrasing your question."
        print(f"An error occurred: {error_msg}")
        return jsonify({"error": error_msg}), 500
    except Exception as e:
        # Generic catch-all for other unexpected errors
        print(f"An error occurred: {e}")
        return jsonify({"error": "An unexpected error occurred. Please check the server logs."}), 500


if __name__ == '__main__':
    app.run(host="0.0.0.0", port= 8080, debug= True)