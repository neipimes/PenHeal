import os
import sys
import json
from typing import Any, Optional
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders import Docx2txtLoader
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import Chroma
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from agent_import import RuntimeModelConfig, create_client



class RAG_module():
    def __init__(
        self,
        document_path,
        embedding_config: Optional[RuntimeModelConfig] = None,
        llm_config: Optional[RuntimeModelConfig] = None,
        chroma_dir: str = "./data",
        rebuild_embeddings: bool = False,
    ):
        # change the path to the document folder
        self.document_path = document_path
        self.chroma_dir = chroma_dir
        self.rebuild_embeddings = rebuild_embeddings

        documents = []
        for file in os.listdir(self.document_path):
            if file.endswith(".pdf"):
                pdf_path = os.path.join(self.document_path, file)
                loader = PyPDFLoader(pdf_path)
                documents.extend(loader.load())
            elif file.endswith('.docx') or file.endswith('.doc'):
                doc_path = os.path.join(self.document_path, file)
                loader = Docx2txtLoader(doc_path)
                documents.extend(loader.load())
            elif file.endswith('.txt'):
                text_path = os.path.join(self.document_path, file)
                loader = TextLoader(text_path)
                documents.extend(loader.load())

        # Split the documents into smaller chunks
        text_splitter = RecursiveCharacterTextSplitter(
            separators=["\n\n", "\n", " ", ""],
            chunk_size=300,
            chunk_overlap=30,
            length_function=len,
        )
        chunks = text_splitter.split_documents(documents)

        # Prepare embeddings
        if embedding_config is None:
            embedding_config = RuntimeModelConfig(is_embedding=True)

        # create embeddings client via factory
        try:
            self.embeddings: Any = create_client(embedding_config, client_type="embeddings")
        except Exception:
            # fallback
            self.embeddings = OpenAIEmbeddings(check_embedding_ctx_length=False)

        # Ensure chroma directory exists
        os.makedirs(self.chroma_dir, exist_ok=True)
        meta_path = os.path.join(self.chroma_dir, "chroma_meta.json")

        # If existing store and not rebuilding, load it and warn if embedding model differs
        if os.path.exists(meta_path) and not self.rebuild_embeddings:
            try:
                with open(meta_path, "r", encoding="utf-8") as mf:
                    meta = json.load(mf)
                stored_model = meta.get("embedding_model")
                current_model = getattr(embedding_config, "model_name", None)
                if current_model and stored_model and current_model != stored_model:
                    print(
                        f"Warning: current embedding model ({current_model}) differs from stored embeddings ({stored_model})."
                        " Set rebuild_embeddings=True to recompute embeddings."
                    )
            except Exception:
                pass

        # Build or load the vector store
        if self.rebuild_embeddings or not os.path.exists(os.path.join(self.chroma_dir, "index")):
            # create new vectordb
            self.vectordb = Chroma.from_documents(chunks, embedding=self.embeddings, persist_directory=self.chroma_dir)
            self.vectordb.persist()
            # write metadata
            try:
                with open(meta_path, "w", encoding="utf-8") as mf:
                    json.dump({"embedding_model": getattr(embedding_config, "model_name", None)}, mf)
            except Exception:
                pass
        else:
            # load existing
            self.vectordb = Chroma(persist_directory=self.chroma_dir, embedding_function=self.embeddings)

        # Store LLM and retriever as separate components (modern LangChain pattern)
        if llm_config is None:
            try:
                self.llm = ChatOpenAI(temperature=0.7, model="gpt-4o")
            except Exception:
                self.llm = None
        else:
            try:
                self.llm = create_client(llm_config, client_type="llm")
            except Exception:
                self.llm = None

        self.retriever = self.vectordb.as_retriever(search_kwargs={"k": 3})

    def add_document(self, document, persist: bool = True):
        # document: list of Document
        if hasattr(self.vectordb, "add_documents"):
            self.vectordb.add_documents(document)
        else:
            # fallback: recreate store (less efficient)
            self.vectordb = Chroma.from_documents(document, embedding=self.embeddings, persist_directory=self.chroma_dir)
        if persist:
            self.vectordb.persist()

    def ask_question(self, query):
        """
        Retrieve relevant documents for the given query.
        Returns a list of source document contents for agents to use.
        
        Args:
            query: The question to search for in the knowledge base
            
        Returns:
            List of source document page contents (strings)
        """
        # Retrieve relevant documents from the vector store
        try:
            source_documents = self.retriever.invoke(query)
        except Exception:
            source_documents = []
        # Return document contents as a list of strings
        return [doc.page_content for doc in source_documents]
