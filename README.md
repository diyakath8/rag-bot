# RAG Q&A Bot

This project implements a Retrieval-Augmented Generation (RAG) pipeline that allows users to ask questions based on a collection of documents and receive grounded answers with source citations.

## Tech Stack
- Python 3.11
- LangChain
- ChromaDB
- Ollama (llama3, nomic-embed-text)
- PyPDF

## Architecture
Documents → Chunking → Embeddings → Vector DB → Retrieval → LLM → Answer

## Chunking Strategy
Used RecursiveCharacterTextSplitter with:
- Chunk size: 500
- Overlap: 100  
This ensures context is preserved across chunks.

## Embedding Model
- nomic-embed-text (via Ollama)
Chosen because:
- Fast local embeddings
- No API dependency

## Vector Database
- ChromaDB (persistent)
Chosen because:
- Lightweight
- Easy local storage
- No re-indexing required

## Setup Instructions

1. Install dependencies:
2. Install Ollama:
https://ollama.com

3. Pull models:
4. Run indexing:

## Example Queries
- What is climate change?
- What is artificial intelligence?
- What is a balanced diet?
- Explain the solar system
- How does climate change affect education?

## Known Limitations
- Performance depends on document quality
- Slide-based PDFs may lack definitions
- Retrieval may miss context if chunk not selected
- Local models are slower than cloud APIs
