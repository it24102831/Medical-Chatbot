# Medical Chatbot

This project is a Retrieval-Augmented Generation (RAG) based medical chatbot. It leverages a private collection of medical documents to provide informed answers to user questions. The application is built with Python, Flask, and LangChain, using Pinecone for vector storage and OpenRouter to access state-of-the-art language models.

## Features

- **RAG Architecture**: Answers questions based on information retrieved from a private collection of medical PDF documents.
- **Web Interface**: A simple and clean chat interface for users to interact with the chatbot.
- **REST API**: A `/get` endpoint to programmatically interact with the chatbot.
- **Efficient Initialization**: The RAG chain is initialized lazily on the first request to ensure fast server startup.
- **Modular Design**: The code is organized into modules for data processing, prompts, and the main application logic.
- **Hardware Acceleration**: Automatically uses CUDA or MPS (for Apple Silicon) for embedding calculations if available, falling back to CPU.
- **Robust Error Handling**: Gracefully handles common issues like invalid API keys and LLM output parsing errors.
- **Extensible**: Easily add more documents to the `data` directory to expand the chatbot's knowledge base.

## How It Works

The chatbot follows a two-stage process:

1.  **Indexing (Data Ingestion)**:
    -   PDF documents from the `data/` directory are loaded.
    -   The documents are split into smaller text chunks.
    -   A pre-trained model from Hugging Face (`sentence-transformers/all-MiniLM-L6-v2`) is used to create vector embeddings for each chunk.
    -   These embeddings are stored and indexed in a Pinecone vector database. This is a one-time process performed by the `store_index.py` script.

2.  **Retrieval and Generation (Inference)**:
    -   When a user asks a question, the application creates an embedding for the question.
    -   It queries the Pinecone index to find the most semantically similar text chunks (the context).
    -   The user's question and the retrieved context are passed to a Large Language Model (LLM) like `openai/gpt-4o-mini` via OpenRouter.
    -   The LLM generates a concise, context-aware answer, which is then sent back to the user.

## Tech Stack

- **Backend**: Python, Flask
- **AI/ML Framework**: LangChain
- **Vector Database**: Pinecone
- **Embeddings Model**: Hugging Face Sentence Transformers
- **LLM Provider**: OpenRouter (accessing models like `openai/gpt-4o-mini`)
- **Environment Management**: `dotenv`, `virtualenv`

## Project Structure

```
Medical-Chatbot/
├── app.py                # Main Flask application with API endpoints
├── store_index.py        # Script to process PDFs and store them in Pinecone
├── src/
│   ├── helper.py         # Helper functions for loading PDFs, splitting text, etc.
│   └── prompt.py         # Contains the system prompt for the LLM
├── templates/
│   └── chat.html         # Frontend HTML for the chat interface
├── venv/                   # Python virtual environment (after setup)
├── data/                 # Directory for your source PDF documents
├── .env                  # File for storing environment variables (API keys)
├── requirements.txt      # Python dependencies
└── README.md             # This file
```

## Setup and Installation

Follow these steps to set up and run the project locally.

### 1. Clone the Repository

```bash
git clone https://github.com/Charindu01/Medical-Chatbot.git
cd Medical-Chatbot
```

### 2. Create and Activate a Virtual Environment

```bash
# For macOS/Linux
python3 -m venv venv
source venv/bin/activate

# For Windows
python -m venv venv
.\venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Set Up Environment Variables

Create a `.env` file in the root directory and add your API keys. You can get these keys from Pinecone and OpenRouter (or OpenAI).

```env
# .env
PINECONE_API_KEY="YOUR_PINECONE_API_KEY"
OPENROUTER_API_KEY="YOUR_OPENROUTER_API_KEY"
INDEX_NAME="medical-chatbot"
```

## Usage

### 1. Add Your Data

Place your medical PDF files inside the `data/` directory.

### 2. Create the Vector Index

Run the `store_index.py` script to process your documents and populate the Pinecone index. This only needs to be done once, or whenever you add new documents.

```bash
python store_index.py
```

### 3. Run the Flask Application

Start the web server.

```bash
python app.py
```

The application will be available at `http://127.0.0.1:8080`. Open this URL in your browser to start chatting.
