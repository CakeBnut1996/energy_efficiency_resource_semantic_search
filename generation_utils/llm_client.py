import os, json
from typing import Type, Any
from dotenv import load_dotenv
from pydantic import BaseModel
import streamlit as st
from generation_utils.schema import Response

# --- IMPORTS FOR PROVIDERS ---
try:
    from google import genai

    HAS_GEMINI = True
except ImportError:
    HAS_GEMINI = False

try:
    from openai import OpenAI

    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False

try:
    import anthropic

    HAS_ANTHROPIC = True
except ImportError:
    HAS_ANTHROPIC = False

try:
    from groq import Groq

    HAS_GROQ = True
except ImportError:
    HAS_GROQ = False


class LLMClient:
    def __init__(self, provider: str, model_name: str):
        load_dotenv()
        self.provider = provider.lower()
        self.model_name = model_name
        self._init_client()
        print(f"🔌 LLM Client connected: {self.provider}/{self.model_name}")

    def _init_client(self):
        # Helper to fetch key from env first, then streamlit secrets
        def get_key(env_name, secret_path):
            return os.getenv(env_name) or st.secrets.get("gkeys", {}).get(secret_path) or st.secrets.get(env_name)

        if self.provider == "gemini":
            if not HAS_GEMINI: raise ImportError("Run `pip install google-genai`")
            api_key = get_key("GEMINI_API_KEY", "gemini")
            self.client = genai.Client(api_key=api_key)

        elif self.provider == "openai":
            if not HAS_OPENAI: raise ImportError("Run `pip install openai`")
            api_key = get_key("OPENAI_API_KEY", "openai")
            self.client = OpenAI(api_key=api_key)

        elif self.provider == "anthropic":
            if not HAS_ANTHROPIC: raise ImportError("Run `pip install anthropic`")
            api_key = get_key("ANTHROPIC_API_KEY", "anthropic")
            self.client = anthropic.Anthropic(api_key=api_key)

        elif self.provider == "groq":
            if not HAS_GROQ: raise ImportError("Run `pip install groq`")
            api_key = get_key("GROQ_API_KEY", "groq")
            self.client = Groq(api_key=api_key)

        else:
            raise ValueError(f"Unknown provider: {self.provider}")

    def generate_text(self, prompt: str, system_instruction: str = None) -> str:
        """Standard text generation for the Student."""
        try:
            # --- GEMINI ---
            if self.provider == "gemini":
                return self.client.models.generate_content(
                    model=self.model_name, contents=prompt
                ).text

            # --- OPENAI ---
            elif self.provider == "openai":
                messages = [{"role": "user", "content": prompt}]
                if system_instruction:
                    messages.insert(0, {"role": "system", "content": system_instruction})
                res = self.client.chat.completions.create(
                    model=self.model_name, messages=messages
                )
                return res.choices[0].message.content

            # --- ANTHROPIC ---
            elif self.provider == "anthropic":
                res = self.client.messages.create(
                    model=self.model_name, max_tokens=1024,
                    messages=[{"role": "user", "content": prompt}]
                )
                return res.content[0].text

            # --- GROQ ---
            elif self.provider == "groq":
                messages = [{"role": "user", "content": prompt}]
                if system_instruction:
                    messages.insert(0, {"role": "system", "content": system_instruction})
                res = self.client.chat.completions.create(
                    model=self.model_name, messages=messages
                )
                return res.choices[0].message.content

        except Exception as e:
            return f"Error: {e}"

    def generate_structured(self, prompt: str, schema_model: Type[BaseModel]) -> Any:
        """Structured output """
        try:
            # --- GEMINI STRUCTURED ---
            if self.provider == "gemini":
                response = self.client.models.generate_content(
                    model=self.model_name,
                    contents=prompt,
                    config={
                        "response_mime_type": "application/json",
                        "response_schema": schema_model
                    }
                )
                return schema_model.model_validate_json(response.text)

            # --- OPENAI STRUCTURED ---
            elif self.provider == "openai":
                completion = self.client.beta.chat.completions.parse(
                    model=self.model_name,
                    messages=[{"role": "user", "content": prompt}],
                    response_format=schema_model
                )
                return completion.choices[0].message.parsed


            elif self.provider == "groq":
                # https://console.groq.com/docs/structured-outputs
                messages = [
                    {"role": "system", "content": "You are a helpful assistant that outputs JSON."},
                    {"role": "user", "content": prompt}
                ]
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                    response_format={
                        "type": "json_schema",
                        "json_schema": {
                            "name": "Response",
                            "schema": Response.model_json_schema()
                        }
                    }
                )
                response_formatted = Response.model_validate(json.loads(response.choices[0].message.content))
                return response_formatted

            else:
                raise NotImplementedError("Structured output only supported for Gemini, Groq, and OpenAI.")

        except Exception as e:
            print(f"⚠️ Structured Generation Error: {e}")
            # Return empty model on failure to prevent crash
            return schema_model.model_construct()