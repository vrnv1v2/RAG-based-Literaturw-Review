import torch
import faiss
import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer, CrossEncoder
from langchain_core.documents import Document

class EmbeddingModel:
    def __init__(self, model_name: str):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = SentenceTransformer(model_name, device=self.device)
        self.query_instruction = "Represent this sentence for searching relevant passages: "

    def embed_documents(self, chunks: list[str]) -> np.ndarray:
        return self.model.encode(chunks, show_progress_bar=False, convert_to_numpy=True, batch_size=32)

    def embed_query(self, query: str) -> np.ndarray:
        return self.model.encode(f"{self.query_instruction}{query}", convert_to_numpy=True)

class FAISSStore:
    def __init__(self):
        self.index = None
        self.doc_map = {}

    def build_store(self, langchain_documents: list[Document], embedding_engine: EmbeddingModel):
        if not langchain_documents: return
        raw_texts = [doc.page_content for doc in langchain_documents]
        vectors = np.array(embedding_engine.embed_documents(raw_texts)).astype('float32')
        faiss.normalize_L2(vectors)
        self.index = faiss.IndexFlatIP(vectors.shape[1])
        self.index.add(vectors)
        for idx, doc in enumerate(langchain_documents):
            self.doc_map[idx] = doc

    def search_and_retrieve(self, query_embedding: np.ndarray, k: int = 4) -> list[Document]:
        f_query = np.array([query_embedding]).astype('float32')
        faiss.normalize_L2(f_query)
        scores, indices = self.index.search(f_query, k=k)
        retrieved_documents = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1: continue
            matched_doc = self.doc_map[idx]
            matched_doc.metadata["search_score"] = float(score)
            retrieved_documents.append(matched_doc)
        return retrieved_documents

class DenseRetriever:
    def __init__(self, faiss_store, embedding_model, threshold: float):
        self.faiss_store = faiss_store
        self.embedding_model = embedding_model
        self.threshold = threshold

    def retrieve(self, query_str: str, k: int = 5):
        query_vector = self.embedding_model.embed_query(query_str)
        matched_docs = self.faiss_store.search_and_retrieve(query_vector, k=k)
        return [{"document": d, "text": d.page_content, "score": d.metadata["search_score"]} 
                for d in matched_docs if d.metadata["search_score"] >= self.threshold]

class BM25Retriever:
    def __init__(self, chunks: list[Document]):
        self.chunks = chunks
        self.model = None

    def build(self):
        tokenized_corpus = [doc.page_content.lower().split(" ") for doc in self.chunks]
        self.model = BM25Okapi(tokenized_corpus)

    def retrieve(self, query_str: str, k: int = 5):
        tokenized_query = query_str.lower().split(" ")
        scores = self.model.get_scores(tokenized_query)
        top_k_indices = np.argsort(scores)[::-1][:k]
        return [{"document": self.chunks[idx], "text": self.chunks[idx].page_content, "score": float(scores[idx])} 
                for idx in top_k_indices if scores[idx] > 0.0]

class HybridRetriever:
    def __init__(self, dense_retriever, bm25_retriever, c: int = 60):
        self.dense_retriever = dense_retriever
        self.bm25_retriever = bm25_retriever
        self.c = c

    def retrieve(self, query_str: str, k: int = 5):
        dense_res = self.dense_retriever.retrieve(query_str, k=k*2)
        sparse_res = self.bm25_retriever.retrieve(query_str, k=k*2)
        
        rrf_registry = {}
        for rank, m in enumerate(dense_res, start=1):
            rrf_registry[m["text"]] = {"document": m["document"], "score": 1.0 / (self.c + rank)}
        for rank, m in enumerate(sparse_res, start=1):
            if m["text"] not in rrf_registry:
                rrf_registry[m["text"]] = {"document": m["document"], "score": 0.0}
            rrf_registry[m["text"]]["score"] += 1.0 / (self.c + rank)
            
        sorted_chunks = sorted(rrf_registry.items(), key=lambda x: x[1]["score"], reverse=True)
        return [{"document": v["document"], "text": t, "score": v["score"]} for t, v in sorted_chunks[:k]]

class Reranker:
    def __init__(self, model_name: str):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = CrossEncoder(model_name, device=self.device)

    def rerank(self, query: str, retrieved_chunks: list, top_k: int = 3):
        if not retrieved_chunks: return []
        pairs = [(query, chunk["text"]) for chunk in retrieved_chunks]
        scores = self.model.predict(pairs)
        for idx, score in enumerate(scores):
            retrieved_chunks[idx]["rerank_score"] = float(score)
        
        sorted_res = sorted(retrieved_chunks, key=lambda x: x["rerank_score"], reverse=True)
        final_docs = []
        for item in sorted_res[:top_k]:
            doc = item["document"]
            doc.metadata["final_rerank_score"] = item["rerank_score"]
            final_docs.append(doc)
        return final_docs

class Generator:
    def __init__(self, model_name: str):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        from transformers import AutoTokenizer, AutoModelForCausalLM
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=torch.float16 if self.device == "cuda" else torch.float32
        ).to(self.device)

    def generate(self, prompt: str, max_new_tokens: int = 256) -> str:
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs, max_new_tokens=max_new_tokens, temperature=0.1, do_sample=False, pad_token_id=self.tokenizer.eos_token_id
            )
        return self.tokenizer.decode(output_ids[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()