from generation_utils.llm_client import LLMClient
from typing import Optional, Type, Union
from pydantic import BaseModel

class StudentGenerator:
    def __init__(self, provider: str, model_name: str):
        """
        The Student uses the LLMClient for text generation.
        """
        self.llm = LLMClient(provider, model_name)

    def generate(self, query: str, context: str, schema: Optional[Type[BaseModel]] = None) -> Union[str, BaseModel]:
        system_instr = "You are a helpful assistant. Answer based strictly on the context provided."
        prompt = f"Context:\n{context}\n\nUser question: {query}\n"

        if schema:
            return self.llm.generate_structured(prompt, schema)

        # Clean up this return statement
        response = self.llm.generate_text(prompt, system_instruction=system_instr)
        if response is None:
            return "Error: The LLM returned no response."
        return response