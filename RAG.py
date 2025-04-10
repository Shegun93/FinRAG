from flask import Flask, request, jsonify
import torch
from transformers import BitsAndBytesConfig, AutoTokenizer, AutoModelForCausalLM
from sentence_transformers import SentenceTransformer
from transformers.utils import is_flash_attn_2_available
from pinecone import Pinecone
import streamlit as st
import os
import dotenv as load_env
load_dotenv()

use_quantization_config = True 
device = "cuda" if torch.cuda.is_available() else "cpu"
embedding_model = SentenceTransformer("all-mpnet-base-v2", device=device)
model_id = "google/gemma-2b-it"
st.title("FinSearch Analyst")

# Define quantization config if needed
if use_quantization_config:
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True
    )
else:
    quantization_config = None

if (is_flash_attn_2_available()) and (torch.cuda.get_device_capability(0)[0] >= 8):
    attn_implementation = "flash_attention_2"
else:
    attn_implementation = "sdpa"
print(f"[INFO] Using attention implementation: {attn_implementation}")
print(f"[INFO] Using model_id: {model_id}")

tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path=model_id)
Gamma_model = AutoModelForCausalLM.from_pretrained(
    pretrained_model_name_or_path=model_id, 
    torch_dtype=torch.float16,
    quantization_config=quantization_config,
    low_cpu_mem_usage=True,
    attn_implementation=attn_implementation,
    device_map="auto"  # Let transformers handle device placement
) 

# Remove this block entirely - not needed for quantized models
# if not use_quantization_config:
#     Gamma_model.to("cuda")

class PineconeRetriever:
    def __init__(self, index_name="datatonic-rags", embedding_model=None):
        Vector_db = os.getenv("PINECONE_API_KEY")
        self.pc = Pinecone(api_key= Vector_db)
        self.index = self.pc.Index(index_name)
        self.embedding_model = embedding_model
    
    def query(self, query: str, top_k: int = 1):
        query_embedding = self.embedding_model.encode(query).tolist()
        results = self.index.query(
            vector=query_embedding,
            top_k=top_k,
            include_metadata=True
        )
        chunks_with_scores = [(match.metadata["text"], match.score) for match in results.matches]
        return chunks_with_scores

retrieval = PineconeRetriever(embedding_model=embedding_model)

st.header("Ask a Question")
query = st.text_input("Enter your query:")
if st.button("Generate Answer"):
    if query:
        with st.spinner("Retrieving and generating response..."):
            top_chunk, score = retrieval.query(query, top_k=1)[0]
            
            prompt = f"""Answer the question based on the context below.
    
    Question: {query}
    Context: {top_chunk}
    Answer:"""
            
            inputs = tokenizer(prompt, return_tensors="pt").to(Gamma_model.device)
            outputs = Gamma_model.generate(
                **inputs, temperature=0.7, max_new_tokens=512,
                do_sample=True, pad_token_id=tokenizer.eos_token_id
            )
            
            full_response = tokenizer.decode(outputs[0], skip_special_tokens=True)
            answer = full_response.replace(prompt, "").strip()

            st.subheader("Generated Answer:")
            st.write(answer)

            st.subheader("Retrieved Context:")
            st.write(top_chunk)

            st.subheader("Relevance Score:")
            st.write(score)

if __name__ == "__main__":
    st.run()