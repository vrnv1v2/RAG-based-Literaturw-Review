import yaml
import argparse
import sys
from src.academic_rag.ingestion import PaperRepositoryPipeline
from src.academic_rag.chunking import AcademicChunkingPipeline
from src.academic_rag.models import (
    EmbeddingModel, FAISSStore, DenseRetriever, 
    BM25Retriever, HybridRetriever, Reranker, Generator
)
from src.academic_rag.graph import LiteratureReviewPipeline

def main():
    parser = argparse.ArgumentParser(description="Run Production Graph Academic RAG")
    parser.add_argument("--query", type=str, required=True, help="Your academic question")
    args = parser.parse_args()

    # Load configurations
    with open("config.yaml", "r") as f:
        cfg = yaml.safe_load(f)

    # 1. Document Ingestion (Changed to point strictly to the local directory key)
    repo_pipeline = PaperRepositoryPipeline(data_dir=cfg["paths"]["data_dir"])
    repo = repo_pipeline.run_ingestion()
    
    # Clean fallback exit mechanism if your local data directory contains no target files
    if not repo:
        print("❌ Pipeline Execution Aborted: Missing or empty source documents folder.")
        sys.exit(1)
        
    # 2. Text Partitioning & Windowing
    print("✂️ Slicing texts into sliding sentence context windows...")
    chunks = AcademicChunkingPipeline(
        paper_repository=repo, 
        sentences_per_chunk=cfg["parameters"]["sentences_per_chunk"],
        overlap=cfg["parameters"]["overlap"]
    ).run_chunking()

    # ==========================================
    # Everything below this line remains exactly as originally designed
    # ==========================================
    print("🚀 Initialising Models & Constructing Vector Spaces...")
    embed_model = EmbeddingModel(cfg["models"]["embedding"])
    
    vector_store = FAISSStore()
    vector_store.build_store(chunks, embed_model)
    
    dense_retriever = DenseRetriever(vector_store, embed_model, cfg["parameters"]["dense_threshold"])
    bm25_retriever = BM25Retriever(chunks)
    bm25_retriever.build()
    
    hybrid_orchestrator = HybridRetriever(dense_retriever, bm25_retriever)
    reranker = Reranker(cfg["models"]["reranker"])
    generator = Generator(cfg["models"]["llm"])

    pipeline = LiteratureReviewPipeline(
        hybrid_retriever=hybrid_orchestrator,
        reranker=reranker,
        generator=generator,
        min_score=cfg["parameters"]["rerank_threshold"],
        max_loops=cfg["parameters"]["max_loops"]
    )

    print(f"\n⚡ Executing LangGraph for Query: '{args.query}'")
    result = pipeline.run(args.query)
    
    print("\n================ OUTPUT ================")
    print(result["generation"])
    print("========================================")

if __name__ == "__main__":
    main()