# RAG-based Document Q&A Bot

## Overview
This project implements a Retrieval-Augmented Generation (RAG) pipeline using LangChain, ChromaDB, and Ollama.

## Features
- Multi-document question answering
- Local embeddings and LLM (no API)
- Source-based grounded answers
- Handles unknown questions safely

## Setup
pip install -r requirements.txt

## Run
python ingest.py
python query.py

## Example Questions
- What is artificial intelligence?
- What is climate change?
- What is a balanced diet?

## Tech Stack
- LangChain
- ChromaDB
- Ollama (llama3, nomic-embed-text)
