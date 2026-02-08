"""
Ollama Integration for LangChain Agents

This module provides integration with local Ollama instances for LLM inference.
Uses langchain-ollama for ChatOllama integration with streaming and tool support.
"""

from typing import Any, Dict, List, Optional, Callable, Generator
from dataclasses import dataclass, field
import httpx
import json
import time
from datetime import datetime
import logging

from langchain_core.language_models import BaseChatModel

logger = logging.getLogger(__name__)


@dataclass
class OllamaConfig:
    """Configuration for Ollama connection."""
    base_url: str = "http://localhost:11434"
    model: str = "llama3.2:latest"
    timeout: float = 300.0  # 5 minutes for model loading
    connect_timeout: float = 10.0  # Separate connect timeout
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 40
    num_ctx: int = 8192
    num_predict: int = 2048
    repeat_penalty: float = 1.1
    stop: List[str] = field(default_factory=list)
    stream: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return {
            "base_url": self.base_url,
            "model": self.model,
            "timeout": self.timeout,
            "stream": self.stream,
            "options": {
                "temperature": self.temperature,
                "top_p": self.top_p,
                "top_k": self.top_k,
                "num_ctx": self.num_ctx,
                "num_predict": self.num_predict,
                "repeat_penalty": self.repeat_penalty,
            }
        }


class OllamaClient:
    """Client for interacting with Ollama API with streaming support."""

    def __init__(self, config: Optional[OllamaConfig] = None):
        self.config = config or OllamaConfig()
        self._timeout = httpx.Timeout(
            connect=self.config.connect_timeout,
            read=self.config.timeout,
            write=30.0,
            pool=10.0
        )

    def _create_client(self) -> httpx.Client:
        """Create a fresh httpx client for each request."""
        return httpx.Client(timeout=self._timeout)

    def check_connection(self) -> bool:
        """Check if Ollama is running and accessible."""
        try:
            with self._create_client() as client:
                response = client.get(f"{self.config.base_url}/api/tags")
                return response.status_code == 200
        except Exception:
            return False

    def list_models(self) -> List[Dict[str, Any]]:
        """List all available models in Ollama."""
        try:
            with self._create_client() as client:
                response = client.get(f"{self.config.base_url}/api/tags")
                if response.status_code == 200:
                    data = response.json()
                    return data.get("models", [])
        except Exception as e:
            logger.warning(f"Error listing models: {e}")
        return []

    def get_running_models(self) -> List[Dict[str, Any]]:
        """Get currently loaded/running models."""
        try:
            with self._create_client() as client:
                response = client.get(f"{self.config.base_url}/api/ps")
                if response.status_code == 200:
                    data = response.json()
                    return data.get("models", [])
        except Exception as e:
            logger.warning(f"Error getting running models: {e}")
        return []

    def is_model_loaded(self) -> bool:
        """Check if the configured model is currently loaded in memory."""
        running = self.get_running_models()
        model_base = self.config.model.split(":")[0]
        for m in running:
            if model_base in m.get("name", ""):
                return True
        return False

    def warmup_model(self, progress_callback: Optional[Callable[[str], None]] = None) -> bool:
        """Pre-load the model into memory with a minimal request."""
        if progress_callback:
            progress_callback("Checking if model is already loaded...")

        if self.is_model_loaded():
            if progress_callback:
                progress_callback("Model already loaded!")
            return True

        if progress_callback:
            progress_callback(f"Loading model {self.config.model} into memory...")

        url = f"{self.config.base_url}/api/generate"
        payload = {
            "model": self.config.model,
            "prompt": "Hi",
            "stream": False,
            "options": {"num_predict": 1}
        }

        try:
            warmup_timeout = httpx.Timeout(connect=30.0, read=600.0, write=30.0, pool=30.0)
            start = time.time()

            with httpx.Client(timeout=warmup_timeout) as client:
                response = client.post(url, json=payload)
                elapsed = time.time() - start

                if response.status_code == 200:
                    if progress_callback:
                        progress_callback(f"Model loaded in {elapsed:.1f}s!")
                    return True
                else:
                    if progress_callback:
                        progress_callback(f"Failed: {response.text}")
                    return False

        except Exception as e:
            if progress_callback:
                progress_callback(f"Error: {e}")
            return False

    def chat_streaming(
        self,
        messages: List[Dict[str, str]],
        token_callback: Optional[Callable[[str], None]] = None
    ) -> Generator[str, None, Dict[str, Any]]:
        """Chat with streaming response."""
        url = f"{self.config.base_url}/api/chat"

        payload = {
            "model": self.config.model,
            "messages": messages,
            "stream": True,
            "options": {
                "temperature": self.config.temperature,
                "top_p": self.config.top_p,
                "top_k": self.config.top_k,
                "num_ctx": self.config.num_ctx,
                "num_predict": self.config.num_predict,
            }
        }

        start_time = time.time()
        tokens_generated = 0

        try:
            client = httpx.Client(timeout=self._timeout)
            response = client.send(client.build_request("POST", url, json=payload), stream=True)

            if response.status_code != 200:
                error_text = response.read().decode()
                client.close()
                raise Exception(f"HTTP {response.status_code}: {error_text}")

            try:
                for line in response.iter_lines():
                    if not line:
                        continue

                    try:
                        data = json.loads(line)
                    except json.JSONDecodeError:
                        continue

                    if "message" in data and "content" in data["message"]:
                        token = data["message"]["content"]
                        tokens_generated += 1

                        if token_callback:
                            token_callback(token)

                        yield token

                    if data.get("done", False):
                        break

            finally:
                response.close()
                client.close()

            total_time = time.time() - start_time
            return {
                "total_time_ms": total_time * 1000,
                "tokens_generated": tokens_generated,
                "tokens_per_second": tokens_generated / total_time if total_time > 0 else 0,
            }

        except httpx.ConnectError as e:
            raise Exception(f"Cannot connect to Ollama at {self.config.base_url}. Is it running? Error: {e}")
        except httpx.TimeoutException as e:
            raise Exception(f"Request timeout: {e}")
        except Exception as e:
            raise


def create_ollama_llm(config: Optional[OllamaConfig] = None) -> BaseChatModel:
    """
    Create a LangChain-compatible Ollama LLM instance.

    Args:
        config: Ollama configuration (uses defaults if not provided)

    Returns:
        ChatOllama instance configured for use with LangChain agents
    """
    config = config or OllamaConfig()

    try:
        from langchain_ollama import ChatOllama

        llm = ChatOllama(
            base_url=config.base_url,
            model=config.model,
            temperature=config.temperature,
            top_p=config.top_p,
            top_k=config.top_k,
            num_ctx=config.num_ctx,
            num_predict=config.num_predict,
            repeat_penalty=config.repeat_penalty,
        )

        logger.info(f"Created ChatOllama with model: {config.model}")
        return llm

    except ImportError:
        try:
            from langchain_community.chat_models import ChatOllama

            llm = ChatOllama(
                base_url=config.base_url,
                model=config.model,
                temperature=config.temperature,
            )

            logger.info(f"Created ChatOllama (community) with model: {config.model}")
            return llm

        except ImportError:
            raise ImportError("Install langchain-ollama: pip install langchain-ollama")


@dataclass
class OllamaMetrics:
    """Metrics for Ollama API calls."""
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    total_tokens_in: int = 0
    total_tokens_out: int = 0
    total_duration_ms: float = 0.0
    request_history: List[Dict[str, Any]] = field(default_factory=list)

    def record_request(
        self,
        success: bool,
        tokens_in: int = 0,
        tokens_out: int = 0,
        duration_ms: float = 0.0,
        model: str = "",
        error: Optional[str] = None
    ):
        self.total_requests += 1
        if success:
            self.successful_requests += 1
        else:
            self.failed_requests += 1

        self.total_tokens_in += tokens_in
        self.total_tokens_out += tokens_out
        self.total_duration_ms += duration_ms

        self.request_history.append({
            "timestamp": datetime.now().isoformat(),
            "success": success,
            "tokens_in": tokens_in,
            "tokens_out": tokens_out,
            "duration_ms": duration_ms,
            "model": model,
            "error": error
        })

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_requests": self.total_requests,
            "successful_requests": self.successful_requests,
            "failed_requests": self.failed_requests,
            "success_rate": (
                self.successful_requests / self.total_requests * 100
                if self.total_requests > 0 else 0
            ),
            "total_tokens_in": self.total_tokens_in,
            "total_tokens_out": self.total_tokens_out,
        }


# Preset configurations
OLLAMA_PRESETS = {
    "default": OllamaConfig(),
    "fast": OllamaConfig(temperature=0.5, num_ctx=4096, num_predict=1024, top_k=20),
    "creative": OllamaConfig(temperature=0.9, top_p=0.95, top_k=60, repeat_penalty=1.05),
    "precise": OllamaConfig(temperature=0.3, top_p=0.7, top_k=10, repeat_penalty=1.2),
    "long_context": OllamaConfig(num_ctx=32768, num_predict=4096),
}


# Connection test utilities
@dataclass
class ConnectionTest:
    """Result of a connection test."""
    success: bool
    latency_ms: float
    endpoint: str
    error: Optional[str] = None
    details: Dict[str, Any] = field(default_factory=dict)


def test_ollama_connection(base_url: str, timeout: float = 10.0) -> ConnectionTest:
    """Test Ollama connection with detailed diagnostics."""
    start_time = time.time()
    endpoint = f"{base_url}/api/tags"

    try:
        with httpx.Client(timeout=timeout) as client:
            response = client.get(endpoint)
            latency = (time.time() - start_time) * 1000

            if response.status_code == 200:
                data = response.json()
                models = data.get("models", [])
                return ConnectionTest(
                    success=True,
                    latency_ms=latency,
                    endpoint=endpoint,
                    details={
                        "status_code": response.status_code,
                        "models_count": len(models),
                        "models": [m.get("name") for m in models[:5]],
                    }
                )
            else:
                return ConnectionTest(
                    success=False,
                    latency_ms=latency,
                    endpoint=endpoint,
                    error=f"HTTP {response.status_code}",
                )

    except httpx.ConnectError as e:
        return ConnectionTest(
            success=False,
            latency_ms=(time.time() - start_time) * 1000,
            endpoint=endpoint,
            error=f"Connection refused - Is Ollama running? ({e})",
        )
    except httpx.TimeoutException as e:
        return ConnectionTest(
            success=False,
            latency_ms=(time.time() - start_time) * 1000,
            endpoint=endpoint,
            error=f"Connection timeout after {timeout}s ({e})",
        )
    except Exception as e:
        return ConnectionTest(
            success=False,
            latency_ms=(time.time() - start_time) * 1000,
            endpoint=endpoint,
            error=f"{type(e).__name__}: {e}",
        )


def test_model_availability(base_url: str, model: str, timeout: float = 10.0) -> ConnectionTest:
    """Test if a specific model is available."""
    start_time = time.time()
    endpoint = f"{base_url}/api/show"

    try:
        with httpx.Client(timeout=timeout) as client:
            response = client.post(endpoint, json={"name": model})
            latency = (time.time() - start_time) * 1000

            if response.status_code == 200:
                data = response.json()
                return ConnectionTest(
                    success=True,
                    latency_ms=latency,
                    endpoint=endpoint,
                    details={
                        "model": model,
                        "parameters": data.get("parameters", "unknown"),
                    }
                )
            else:
                return ConnectionTest(
                    success=False,
                    latency_ms=latency,
                    endpoint=endpoint,
                    error=f"Model '{model}' not found (HTTP {response.status_code})",
                )

    except Exception as e:
        return ConnectionTest(
            success=False,
            latency_ms=(time.time() - start_time) * 1000,
            endpoint=endpoint,
            error=f"{type(e).__name__}: {e}",
        )


def test_model_inference(base_url: str, model: str, timeout: float = 300.0) -> ConnectionTest:
    """
    Test model inference with a simple prompt.

    Note: timeout defaults to 300s (5 min) because models may need to be loaded
    into memory on first request, which can take several minutes.
    """
    start_time = time.time()
    endpoint = f"{base_url}/api/generate"

    try:
        # Use a longer timeout for model loading
        inference_timeout = httpx.Timeout(connect=30.0, read=timeout, write=30.0, pool=30.0)

        with httpx.Client(timeout=inference_timeout) as client:
            response = client.post(
                endpoint,
                json={
                    "model": model,
                    "prompt": "Say 'hello' and nothing else.",
                    "stream": False,
                    "options": {"num_predict": 10}
                }
            )
            latency = (time.time() - start_time) * 1000

            if response.status_code == 200:
                data = response.json()
                return ConnectionTest(
                    success=True,
                    latency_ms=latency,
                    endpoint=endpoint,
                    details={
                        "model": model,
                        "response": data.get("response", "")[:100],
                        "eval_count": data.get("eval_count", 0),
                        "load_duration_ms": data.get("load_duration", 0) / 1_000_000,  # ns to ms
                    }
                )
            else:
                return ConnectionTest(
                    success=False,
                    latency_ms=latency,
                    endpoint=endpoint,
                    error=f"Inference failed (HTTP {response.status_code}): {response.text[:200]}",
                )

    except httpx.TimeoutException:
        elapsed = (time.time() - start_time)
        return ConnectionTest(
            success=False,
            latency_ms=elapsed * 1000,
            endpoint=endpoint,
            error=f"Inference timeout after {elapsed:.0f}s - model may still be loading. Try 'Load Model' button first.",
        )
    except Exception as e:
        return ConnectionTest(
            success=False,
            latency_ms=(time.time() - start_time) * 1000,
            endpoint=endpoint,
            error=f"{type(e).__name__}: {e}",
        )
