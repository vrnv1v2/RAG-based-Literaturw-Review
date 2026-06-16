import re
from langchain_core.documents import Document

class SentenceChunker:
    def __init__(self):
        self.sentence_endings = re.compile(r'(?<=[.!?])\s+')

    def chunk(self, text: str) -> list[str]:
        if not text.strip():
            return []
        return [s.strip() for s in self.sentence_endings.split(text) if s.strip()]

class AcademicChunkingPipeline:
    def __init__(self, paper_repository: dict, sentences_per_chunk: int = 4, overlap: int = 1):
        self.repository = paper_repository
        self.chunker = SentenceChunker()
        self.sentences_per_chunk = sentences_per_chunk
        self.overlap = overlap

    def run_chunking(self) -> list[Document]:
        all_chunks = []
        for paper_key, paper_data in self.repository.items():
            raw_text = paper_data["raw_text"]
            file_name = paper_data["file_name"]
            if not raw_text.strip():
                continue

            sentences = self.chunker.chunk(raw_text)
            paper_chunks = []
            i = 0
            while i < len(sentences):
                window_sentences = sentences[i : i + self.sentences_per_chunk]
                doc = Document(
                    page_content=" ".join(window_sentences),
                    metadata={
                        "source": file_name,
                        "chunk_index": len(paper_chunks),
                        "sentence_count": len(window_sentences)
                    }
                )
                paper_chunks.append(doc)
                i += (self.sentences_per_chunk - self.overlap)
            all_chunks.extend(paper_chunks)
        return all_chunks