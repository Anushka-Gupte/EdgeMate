"""Ollama model abstraction and client interface for EdgeMate."""

from __future__ import annotations

import os
import time
from typing import Any, Dict, List, Optional

try:
    import ollama
except ImportError:
    ollama = None


class ModelClient:
    """Abstraction layer for local LLM inference via Ollama or fallback providers."""

    def __init__(
        self,
        model_name: Optional[str] = None,
        host: Optional[str] = None,
    ):
        self.model_name = model_name or os.getenv("MODEL_NAME", "qwen2.5-coder:3b")
        self.host = host or os.getenv("OLLAMA_HOST", "http://localhost:11434")
        self._client = None
        self._init_client()

    def _init_client(self) -> None:
        if ollama is not None:
            try:
                self._client = ollama.Client(host=self.host)
            except Exception:
                self._client = None

    def check_connection(self) -> Dict[str, Any]:
        """Check if Ollama service is reachable and which models are installed."""
        if ollama is None:
            return {
                "connected": False,
                "error": "The 'ollama' Python library is not installed.",
                "models": [],
            }
        try:
            client = self._client or ollama.Client(host=self.host)
            response = client.list()
            # Extract model names
            model_names = []
            models_list = response.models if hasattr(response, "models") else response.get("models", [])
            for m in models_list:
                name = m.model if hasattr(m, "model") else m.get("name", "")
                if name:
                    model_names.append(name)

            has_target = any(
                self.model_name in name or name in self.model_name
                for name in model_names
            )

            return {
                "connected": True,
                "error": None,
                "models": model_names,
                "target_model_installed": has_target,
                "target_model": self.model_name,
            }
        except Exception as e:
            return {
                "connected": False,
                "error": f"Ollama isn't running or reachable at {self.host}. Start Ollama and try again. Details: {str(e)[:100]}",
                "models": [],
                "target_model_installed": False,
                "target_model": self.model_name,
            }

    def generate(
        self,
        prompt: str,
        system: Optional[str] = None,
        history: Optional[List[Dict[str, str]]] = None,
        temperature: float = 0.6,
        max_tokens: int = 256,
    ) -> str:
        """Generate response from the configured local model.

        Args:
            prompt: Main user/task prompt.
            system: System instructions.
            history: Optional conversation message history [{"role": "user"|"assistant", "content": ...}].
            temperature: Sampling temperature.
            max_tokens: Maximum tokens to generate.

        Returns:
            Generated response string.
        """
        status = self.check_connection()
        if not status["connected"]:
            return f"⚠️ {status['error']}"

        # If target model is not installed but other models exist, try using an available one or prompt pull
        active_model = self.model_name
        if not status["target_model_installed"] and status["models"]:
            # Pick first available text model if target is missing
            for m in status["models"]:
                if "embed" not in m:
                    active_model = m
                    break

        messages: List[Dict[str, str]] = []
        if system:
            messages.append({"role": "system", "content": system})

        if history:
            for msg in history:
                role = "assistant" if msg.get("role") in ["interviewer", "assistant"] else "user"
                content = msg.get("content", "")
                if content:
                    messages.append({"role": role, "content": content})

        messages.append({"role": "user", "content": prompt})

        try:
            client = self._client or ollama.Client(host=self.host)
            response = client.chat(
                model=active_model,
                messages=messages,
                options={
                    "temperature": temperature,
                    "num_predict": max_tokens,
                },
            )
            content = response.message.content if hasattr(response, "message") else response.get("message", {}).get("content", "")
            return content.strip()
        except Exception as e:
            err_msg = str(e)
            if "not found" in err_msg.lower():
                return f"⚠️ The configured model '{self.model_name}' is not installed in Ollama. Run: `ollama pull {self.model_name}` first."
            return f"⚠️ Error communicating with Ollama: {err_msg}"

    def benchmark_model(
        self,
        model_name: str,
        prompt: str,
        system: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Run a benchmark test on a specific model to evaluate latency and response."""
        start_time = time.time()
        try:
            client = self._client or ollama.Client(host=self.host)
            messages = []
            if system:
                messages.append({"role": "system", "content": system})
            messages.append({"role": "user", "content": prompt})

            response = client.chat(
                model=model_name,
                messages=messages,
                options={"temperature": 0.3, "num_predict": 128},
            )
            elapsed_ms = int((time.time() - start_time) * 1000)
            text = response.message.content if hasattr(response, "message") else response.get("message", {}).get("content", "")
            word_count = len(text.split())

            return {
                "model": model_name,
                "latency_ms": elapsed_ms,
                "word_count": word_count,
                "response": text.strip(),
                "success": True,
                "error": None,
            }
        except Exception as e:
            elapsed_ms = int((time.time() - start_time) * 1000)
            return {
                "model": model_name,
                "latency_ms": elapsed_ms,
                "word_count": 0,
                "response": "",
                "success": False,
                "error": str(e),
            }


_global_model_client: Optional[ModelClient] = None


def get_model_client(model_name: Optional[str] = None) -> ModelClient:
    """Singleton getter for model client."""
    global _global_model_client
    if _global_model_client is None or (model_name and _global_model_client.model_name != model_name):
        _global_model_client = ModelClient(model_name=model_name)
    return _global_model_client
