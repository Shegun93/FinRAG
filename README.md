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

## Setup Instructions
- Prerequisites
- Python 3.9+
- GPU with sufficient VRAM (minimum 5GB recommended)
- Pinecone API key
- Hugging Face account (for Gemma model access)

## 🛠️ Setup

```bash
git clone https://github.com/Shegun93/FinRAG.git
cd FinRAG
pip install -r requirements.txt
```
## 🔑 Configuration

Create a .env configuration file
```bash
PINECONE_API_KEY = "your-api-key"
PINECONE_ENVIRONMENT="region"
```
## Authenticate Hugging Face:
```
huggingface-cli login
```
## Usage
```
# Example query
query = "What was the operating profit increase from 2011-2012?"
answer = ask(query)
print(answer)
```
## The system will:
- Retrieve the most relevant document chunks
- Generate accurate answers using the Gemma LLM
- Display both the answer and the context used

## Customization
- Chunk size: Adjust the chunk_size parameter in the split_into_chunks function
- Model selection: The system automatically selects the appropriate Gemma model based on available GPU memory
- Temperature: Control answer creativity via the temperature parameter in the ask function

## Troubleshooting
- GPU Memory Errors: If you encounter memory issues, try:
- Using the 2B model instead of 7B
- Enabling 4-bit quantization
- Reducing the max_new_tokens parameter

## Pinecone Issues: Ensure your:
- API key is correct
- Index name is unique
- Region matches your Pinecone configuration

License
This project is licensed under the MIT License - see the LICENSE file for details.

**To use this:**
1. Copy the entire markdown above
2. Paste into a new `README.md` file in your repo
3. The badge logos will render automatically on GitHub
4. Customize the URLs and details as needed

The badges use Shields.io formatting which GitHub renders natively - no image files required!

Acknowledgments
Pinecone for the vector database
Hugging Face for the Transformers library and model hub

Google for the Gemma models
