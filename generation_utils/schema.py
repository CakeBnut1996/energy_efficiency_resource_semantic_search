from pydantic import BaseModel, Field, model_validator
from typing import List, Optional

class DatasetSummary(BaseModel):
    """Summarizes a specific dataset and its most relevant evidence."""
    name: Optional[str] = Field(None,description="dataset_id from the retrieval")
    summary: Optional[str] = Field(None,description="A summary of the dataset content relevant to the query based on its returned chunks")
    quote: Optional[str] = Field(None,description="Top-ranked chunk quoted from the dataset")

class Response(BaseModel):
    """The structured final response containing the answer and supporting evidence."""
    answer: Optional[str] = Field(None,description="A concise and factual answer to the user's question based on retrieval")
    name_top: Optional[str] = Field(None,description="dataset_id from the top retrieval")
    supporting_datasets: List[DatasetSummary] = Field(
        default_factory=list,
        description="Supporting evidence list"
    )