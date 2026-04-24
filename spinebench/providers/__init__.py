"""Model providers. One narrow interface, several backends."""

from spinebench.providers.base import ChatProvider, ProviderError
from spinebench.providers.hf_inference import HFInferenceProvider

__all__ = ["ChatProvider", "HFInferenceProvider", "ProviderError"]
