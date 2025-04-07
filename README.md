# FinRAG
# Financial RAG System with Pinecone and Gemma 💰🔍

## Overview

This project implements a Retrieval-Augmented Generation (RAG) system for financial documents, combining vector search with a large language model to provide accurate answers to financial queries.

## 🛠️ Technologies & Tools

| Category       | Technologies                                                                 |
|----------------|-----------------------------------------------------------------------------|
| **Language**   | <img src="https://img.shields.io/badge/Python-3776AB?logo=python&logoColor=white" height="20"> |
| **Vector DB**  | <img src="https://img.shields.io/badge/Pinecone-430098?logo=pinecone&logoColor=white" height="20"> |
| **NLP**        | <img src="https://img.shields.io/badge/spaCy-09A3D5?logo=spacy&logoColor=white" height="20"> |
| **ML**         | <img src="https://img.shields.io/badge/PyTorch-EE4C2C?logo=pytorch&logoColor=white" height="20"> |
| **LLM**        | <img src="https://img.shields.io/badge/Gemma-FFD166?logo=google&logoColor=white" height="20"> |
| **Framework**  | <img src="https://img.shields.io/badge/Hugging%20Face-FFD21F?logo=huggingface&logoColor=black" height="20"> |
| **Data**       | <img src="https://img.shields.io/badge/Pandas-150458?logo=pandas&logoColor=white" height="20"> |

## 🚀 Features

- 📄 Document processing and chunking
- 🔢 Vector embedding generation
- ⚡ Efficient vector storage/retrieval
- 💡 Context-aware question answering
- 🎚️ Automatic model selection based on GPU

## 🛠️ Setup

```bash
git clone https://github.com/yourusername/financial-rag-system.git
cd financial-rag-system
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
