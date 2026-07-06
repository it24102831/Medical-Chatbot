from dotenv import load_dotenv
import os
from src.helper import load_pdf_files, text_split, download_hugging_face_embeddings
from pinecone import Pinecone, ServerlessSpec
from pinecone.core.client.exceptions import UnauthorizedException
from langchain_pinecone import PineconeVectorStore

load_dotenv()

PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY')


# Check for the essential API key
if not PINECONE_API_KEY or 'your' in PINECONE_API_KEY.lower():
    raise ValueError("PINECONE_API_KEY is missing or appears to be a placeholder in your .env file. "
                     "Please replace it with your actual key from the Pinecone dashboard.")

# Configuration
INDEX_NAME = os.environ.get('INDEX_NAME', 'medical-chatbot')
EMBEDDING_DIMENSION = 384
PINECONE_METRIC = "cosine"
PINECONE_CLOUD = "aws"
PINECONE_REGION = "us-east-1"

# 1. Load and process documents
print("Loading PDF documents...")
extracted_data = load_pdf_files(data='data/')
print(f"Loaded {len(extracted_data)} documents.")
text_chunks = text_split(extracted_data)
print(f"Split documents into {len(text_chunks)} chunks.")

# 2. Initialize embeddings model
print("Downloading embeddings model...")
embeddings = download_hugging_face_embeddings()

# 3. Initialize Pinecone and verify connection
try:
    pc = Pinecone(api_key=PINECONE_API_KEY)
    
    # 4. Create or connect to the index by checking if it exists
    if INDEX_NAME not in pc.list_indexes().names():
        print(f"Creating new index: {INDEX_NAME}")
        pc.create_index(
            name=INDEX_NAME,
            dimension=EMBEDDING_DIMENSION,
            metric=PINECONE_METRIC,
            spec=ServerlessSpec(cloud=PINECONE_CLOUD, region=PINECONE_REGION),
        )
    else:
        print(f"Index '{INDEX_NAME}' already exists.")
    
    # 5. Upsert documents into the index
    print("Upserting documents into Pinecone index...")
    PineconeVectorStore.from_documents(
        documents=text_chunks,
        index_name=INDEX_NAME,
        embedding=embeddings,
    )
    print("Data ingestion complete.")
except UnauthorizedException:
    print("Authentication failed. Please check if your PINECONE_API_KEY in the .env file is correct and valid.")
except Exception as e:
    print(f"An unexpected error occurred: {e}")