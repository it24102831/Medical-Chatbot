from typing import List

from langchain_core.documents import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader # type: ignore
from langchain_huggingface import HuggingFaceEmbeddings
import torch


def load_pdf_files(data):
    loader = DirectoryLoader(
        data,
        glob="*.pdf",
        loader_cls=PyPDFLoader,
    )
    return loader.load()


def text_split(docs: List[Document]):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100,
    )
    return text_splitter.split_documents(docs)


def download_hugging_face_embeddings():
    """Downloads embeddings from Hugging Face."""
    def get_device():
        """Checks for and returns the best available hardware for torch."""
        if torch.cuda.is_available():
            return "cuda"
        # Check for Apple Silicon MPS
        if torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    device = get_device()
    print(f"Embeddings model will use device: {device}")
    # Use the recommended HuggingFaceEmbeddings from langchain_huggingface
    return HuggingFaceEmbeddings(model_name=model_name, model_kwargs={'device': device})