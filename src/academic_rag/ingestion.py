import os
import pymupdf

class AcademicPaperReader:
    """Handles layout-aware structural text extraction from individual PDF files."""
    def __init__(self, file_path: str):
        self.file_path = file_path
        self.file_name = os.path.basename(file_path)

    def extract_text(self) -> str:
        try:
            doc = pymupdf.open(self.file_path)
            structured_blocks = []
            for page in doc:
                blocks = page.get_text("blocks")
                for block in blocks:
                    text_content = block[4].strip()
                    if text_content:
                        structured_blocks.append(text_content)
            doc.close()
            return "\n\n".join(structured_blocks)
        except Exception as e:
            print(f"❌ Error reading data from {self.file_name}: {str(e)}")
            return ""


class PaperRepositoryPipeline:
    """Manages collection, loading, parsing, and storing documents locally."""
    def __init__(self, data_dir: str):
        # Changed from hardcoded cloud path to target the config-driven local data directory
        self.base_dir = data_dir

    def run_ingestion(self) -> dict:
        # Failsafe: Create the directory automatically if it doesn't exist yet
        if not os.path.exists(self.base_dir):
            os.makedirs(self.base_dir, exist_ok=True)
            print(f"📁 Created local landing directory at: {self.base_dir}")
            print("💡 Action Required: Please place your 13 academic PDFs into the root './data' folder.")
            return {}

        pdf_files = [f for f in os.listdir(self.base_dir) if f.lower().endswith('.pdf')]
        
        if not pdf_files:
            print(f"⚠️ No PDF files discovered inside '{self.base_dir}'. Ingestion skipped.")
            return {}

        paper_repository = {}

        for file_name in pdf_files:
            paper_key = os.path.splitext(file_name)[0]
            full_path = os.path.join(self.base_dir, file_name)

            reader = AcademicPaperReader(full_path)
            extracted_text = reader.extract_text()

            paper_repository[paper_key] = {
                "file_name": file_name,
                "full_path": full_path,
                "raw_text": extracted_text
            }

        print(f"\n✅ Pipeline complete. Processed {len(paper_repository)} entities locally.")
        return paper_repository