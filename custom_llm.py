# custom_llm.py

import requests
import asyncio
from typing import List, Any, Optional

from pydantic import Field
from llama_index.core.llms import LLM, ChatResponse, ChatMessage, LLMMetadata


class CustomOllamaLLM(LLM):
    model_name: str = Field(...)
    base_url: str = Field(..., description="Base URL for Ollama server root")
    timeout: int = 60

    class Config:
        arbitrary_types_allowed = True

    @property
    def metadata(self) -> LLMMetadata:
        return LLMMetadata(
            context_window=4096,
            num_output=512,
            is_chat_model=True,
            model_name=self.model_name
        )

    def complete(self, prompt: str, stop: Optional[List[str]] = None, **kwargs) -> ChatResponse:
        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "stream": False
        }
        if stop:
            payload["stop"] = stop

        resp = requests.post(self.base_url, json=payload, timeout=self.timeout)
        resp.raise_for_status()
        j = resp.json()
        text = j.get("response") or j.get("text") or j.get("output") or ""

        return ChatResponse(message=ChatMessage(role="assistant", content=text))

    def chat(self, messages: List[Any], **kwargs) -> ChatResponse:
        prompt = "\n".join(f"{m.role}: {m.content}" for m in messages)
        return self.complete(prompt, **kwargs)

    # ✅ Required abstract methods implemented
    def stream_complete(self, prompt: str, **kwargs):
        raise NotImplementedError("stream_complete is not implemented for CustomOllamaLLM")

    def stream_chat(self, messages: List[Any], **kwargs):
        raise NotImplementedError("stream_chat is not implemented for CustomOllamaLLM")

    async def acomplete(self, prompt: str, stop: Optional[List[str]] = None, **kwargs) -> ChatResponse:
        return await asyncio.to_thread(self.complete, prompt, stop, **kwargs)

    async def achat(self, messages: List[Any], **kwargs) -> ChatResponse:
        return await asyncio.to_thread(self.chat, messages, **kwargs)

    async def astream_complete(self, prompt: str, **kwargs):
        raise NotImplementedError("astream_complete is not implemented for CustomOllamaLLM")

    async def astream_chat(self, messages: List[Any], **kwargs):
        raise NotImplementedError("astream_chat is not implemented for CustomOllamaLLM")
