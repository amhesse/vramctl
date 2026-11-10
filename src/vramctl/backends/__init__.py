"""Backend adapters for Ollama, llama.cpp, and Strata."""

from vramctl.backends.base import BackendError
from vramctl.backends.llamacpp import LlamaCppBackend
from vramctl.backends.ollama import OllamaBackend
from vramctl.backends.strata import StrataBackend

__all__ = ["BackendError", "LlamaCppBackend", "OllamaBackend", "StrataBackend"]
