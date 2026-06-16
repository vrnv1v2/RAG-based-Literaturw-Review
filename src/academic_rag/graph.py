from langgraph.graph import StateGraph, START, END
from .state import AdvancedProductionState

class LiteratureReviewPipeline:
    def __init__(self, hybrid_retriever, reranker, generator, min_score: float, max_loops: int):
        self.retriever = hybrid_retriever
        self.reranker = reranker
        self.generator = generator
        self.min_score = min_score
        self.max_loops = max_loops
        self.workflow = self._compile_graph()

    def retrieve_and_rank_node(self, state: AdvancedProductionState) -> dict:
        hybrid_candidates = self.retriever.retrieve(state["question"], k=6)
        if not hybrid_candidates:
            return {"pristine_documents": [], "pipeline_status": "fallback"}
        
        final_docs = self.reranker.rerank(query=state["question"], retrieved_chunks=hybrid_candidates, top_k=3)
        if not final_docs or final_docs[0].metadata.get("final_rerank_score", -99.0) < self.min_score:
            return {"pristine_documents": [], "pipeline_status": "fallback"}
            
        return {"pristine_documents": final_docs, "pipeline_status": "normal"}

    def generate_answer_node(self, state: AdvancedProductionState) -> dict:
        docs = state.get("pristine_documents", [])
        context_str = "\n\n".join([f"[Context {i+1}]: {d.page_content}" for i, d in enumerate(docs)])
        prompt = (
            "<|system|>\n"
            "You are a strict, precise academic assistant. Answer using ONLY the facts explicitly stated below.\n"
            f"--- START OF SOURCE DOCUMENTS ---\n{context_str}\n--- END OF SOURCE DOCUMENTS ---\n"
            f"<|user|>\n{state['question']}\n<|assistant|>\n"
        )
        return {"generation": self.generator.generate(prompt)}

    def query_reformer_node(self, state: AdvancedProductionState) -> dict:
        prompt = f"Provide alternative academic search terms for: {state['question']}. Output ONLY terms."
        upgraded_query = self.generator.generate(prompt, max_new_tokens=50)
        return {"question": upgraded_query.strip(), "loop_count": state["loop_count"] + 1}

    def fallback_node(self, state: AdvancedProductionState) -> dict:
        return {"generation": "I cannot find the answer in the provided text."}

    def route_after_retrieved(self, state: AdvancedProductionState) -> str:
        if state["pipeline_status"] == "fallback":
            return "rewrite" if state["loop_count"] < self.max_loops else "fallback"
        return "generate"

    def _compile_graph(self):
        builder = StateGraph(AdvancedProductionState)
        builder.add_node("retrieve_and_rank", self.retrieve_and_rank_node)
        builder.add_node("generate_answer", self.generate_answer_node)
        builder.add_node("query_reformer", self.query_reformer_node)
        builder.add_node("fallback_handler", self.fallback_node)

        builder.add_edge(START, "retrieve_and_rank")
        builder.add_conditional_edges(
            "retrieve_and_rank",
            self.route_after_retrieved,
            {"rewrite": "query_reformer", "fallback": "fallback_handler", "generate": "generate_answer"}
        )
        builder.add_edge("query_reformer", "retrieve_and_rank")
        builder.add_edge("generate_answer", END)
        builder.add_edge("fallback_handler", END)
        return builder.compile()

    def run(self, user_question: str) -> dict:
        return self.workflow.invoke({
            "question": user_question, "pristine_documents": [], 
            "generation": "", "pipeline_status": "normal", "loop_count": 0
        })