import operator
from typing import List, TypedDict, Annotated
from langchain_core.documents import Document
from pydantic import BaseModel, Field

class AdvancedProductionState(TypedDict):
    question: str
    generation: str
    pipeline_status: str
    loop_count: int
    pristine_documents: List[Document]
    accumulated_context: Annotated[List[Document], operator.add]

class DocumentRelevanceVetter(BaseModel):
    is_relevant: bool = Field(description="True if the document chunk provides explicit factual value.")
    confidence_rationale: str = Field(description="Brief sentence explaining validation choice.")