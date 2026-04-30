import os
import sys
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders import Docx2txtLoader
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import Chroma
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings



class RAG_module():
    def __init__(self, document_path):
        # change the path to the document folder
        self.document_path = document_path
        documents = []
        for file in os.listdir(self.document_path):
            if file.endswith(".pdf"):
                pdf_path = "./docs/" + file
                loader = PyPDFLoader(pdf_path)
                documents.extend(loader.load())
            elif file.endswith('.docx') or file.endswith('.doc'):
                doc_path = "./docs/" + file
                loader = Docx2txtLoader(doc_path)
                documents.extend(loader.load())
            elif file.endswith('.txt'):
                text_path = "./docs/" + file
                loader = TextLoader(text_path)
                documents.extend(loader.load())
        
        # Split the documents into smaller chunks
        text_splitter = RecursiveCharacterTextSplitter(
            separators=["\n\n", "\n", " ", ""],
            chunk_size=300,
            chunk_overlap=30,
            length_function=len
        )
        chunks = text_splitter.split_documents(documents)

        # Store embeddings for use in add_document
        self.openai_embeddings = OpenAIEmbeddings()

        # Convert the document chunks to embedding and save them to the vector store
        self.vectordb = Chroma.from_documents(chunks, embedding=self.openai_embeddings, persist_directory="./data")
        self.vectordb.persist()
        
        # Store LLM and retriever as separate components (modern LangChain pattern)
        self.openai_llm = ChatOpenAI(temperature=0.7, model='gpt-4o')
        self.retriever = self.vectordb.as_retriever(search_kwargs={'k': 3})

    def add_document(self, document):
        self.vectordb = Chroma.from_documents(document, embedding=self.openai_embeddings, persist_directory="./data")
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
        source_documents = self.retriever.invoke(query)
        # Return document contents as a list of strings
        return [doc.page_content for doc in source_documents]
