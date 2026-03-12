"""
rag_pipeline.py — Retrieval-Augmented Generation (RAG) Pipeline

This module handles:
  1. Document ingestion: load text files, split into chunks
  2. Embedding: convert chunks to vectors via OpenAI embeddings
  3. Indexing: store vectors in a FAISS index
  4. Retrieval: find top-K most relevant chunks for a query

Analogy: This is like building a searchable index of your company's
internal docs. When an agent needs to "look something up in the docs",
it calls retrieve() and gets the most relevant passages back.

Usage:
    rag = RAGPipeline()
    rag.ingest("path/to/knowledge_base.txt")
    results = rag.retrieve("best practices for LLM deployment")
"""

import os
from typing import List

from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS


class RAGPipeline:
    """
    A simple RAG pipeline backed by FAISS for local vector similarity search.

    Design decisions:
    - FAISS: in-memory, no external service, perfect for demos and development.
             For production, swap with Pinecone or Chroma by changing 1 line.
    - RecursiveCharacterTextSplitter: preserves semantic coherence better than
      fixed-size splitting because it respects paragraphs and sentences.
    - Overlap: 100-token overlap between chunks prevents context loss at boundaries.
    """

    def __init__(
        self,
        embedding_model: str = "text-embedding-3-small",
        chunk_size: int = 500,
        chunk_overlap: int = 100,
        top_k: int = 3,
    ):
        """
        Args:
            embedding_model: OpenAI embedding model name.
            chunk_size:       Target size of each document chunk (in characters).
            chunk_overlap:    Number of characters to overlap between consecutive chunks.
            top_k:            Number of chunks to return per retrieval query.
        """
        self.top_k = top_k
        self.embeddings = OpenAIEmbeddings(model=embedding_model)
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ".", " ", ""],
        )
        self.vector_store: FAISS | None = None
        self._ingested_files: List[str] = []

    # ------------------------------------------------------------------
    # Ingestion
    # ------------------------------------------------------------------

    def ingest(self, file_path: str) -> int:
        """
        Load a text file, split it into chunks, embed, and index in FAISS.

        Args:
            file_path: Path to the text file to ingest.

        Returns:
            Number of document chunks indexed.

        Raises:
            FileNotFoundError: If file_path does not exist.
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Knowledge base file not found: {file_path}")

        # Load the document
        loader = TextLoader(file_path, encoding="utf-8")
        raw_documents = loader.load()

        # Split into chunks
        chunks = self.text_splitter.split_documents(raw_documents)

        # Build or extend the FAISS index
        if self.vector_store is None:
            self.vector_store = FAISS.from_documents(chunks, self.embeddings)
        else:
            # Add new documents to the existing index
            new_store = FAISS.from_documents(chunks, self.embeddings)
            self.vector_store.merge_from(new_store)

        self._ingested_files.append(file_path)
        print(f"[RAG] Indexed {len(chunks)} chunks from: {file_path}")
        return len(chunks)

    def ingest_text(self, text: str, source: str = "in-memory") -> int:
        """
        Ingest raw text directly (useful for testing or dynamic content).

        Args:
            text:   The raw text content to index.
            source: A label identifying the source (for metadata).

        Returns:
            Number of chunks indexed.
        """
        from langchain.schema import Document
        doc = Document(page_content=text, metadata={"source": source})
        chunks = self.text_splitter.split_documents([doc])

        if self.vector_store is None:
            self.vector_store = FAISS.from_documents(chunks, self.embeddings)
        else:
            new_store = FAISS.from_documents(chunks, self.embeddings)
            self.vector_store.merge_from(new_store)

        return len(chunks)

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------

    def retrieve(self, query: str) -> str:
        """
        Find the top-K most semantically similar chunks for a query.

        Args:
            query: The natural-language question or topic to search for.

        Returns:
            Formatted string of retrieved passages with their sources.
        """
        if self.vector_store is None:
            return (
                "[RAG] No documents indexed. "
                "Call ingest() with a knowledge base file first."
            )

        docs = self.vector_store.similarity_search(query, k=self.top_k)

        if not docs:
            return f"[RAG] No relevant documents found for: '{query}'"

        results = []
        for i, doc in enumerate(docs, 1):
            source = doc.metadata.get("source", "unknown")
            content = doc.page_content.strip()
            results.append(f"[Chunk {i} | Source: {source}]\n{content}")

        return "\n\n---\n\n".join(results)

    def retrieve_with_scores(self, query: str) -> List[tuple]:
        """
        Retrieve top-K chunks along with their similarity scores.

        Returns a list of (document, score) tuples.
        Useful for debugging retrieval quality.
        """
        if self.vector_store is None:
            return []
        return self.vector_store.similarity_search_with_score(query, k=self.top_k)

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    @property
    def is_ready(self) -> bool:
        """True if the pipeline has been initialized with at least one document."""
        return self.vector_store is not None

    def __repr__(self) -> str:
        status = f"{len(self._ingested_files)} files" if self._ingested_files else "empty"
        return f"RAGPipeline(top_k={self.top_k}, store={status})"
