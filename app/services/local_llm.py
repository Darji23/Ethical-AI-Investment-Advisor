"""
Local open-weights LLM from Hugging Face — no API key required.

Used as the last-resort tier when Fireworks is unreachable, and as the only tier
when no FIREWORKS_API_KEY is configured. Weights are downloaded from the public
Hugging Face Hub on first use and cached in ~/.cache/huggingface.

Backends (LOCAL_LLM_BACKEND):
  - "mlx":          Apple Silicon via mlx-lm (4-bit quantised, fast, ~0.9 GB for 1.5B)
  - "transformers": any platform via transformers + torch (CUDA / MPS / CPU)
  - "auto":         mlx on macOS arm64 when mlx-lm is installed, else transformers
"""
import platform
import sys
import threading
from typing import Optional

from app.core.config import LOCAL_LLM_BACKEND, LOCAL_LLM_MODEL, LOCAL_LLM_MAX_NEW_TOKENS

DEFAULT_MODELS = {
    "mlx": "mlx-community/Qwen2.5-1.5B-Instruct-4bit",
    "transformers": "Qwen/Qwen2.5-1.5B-Instruct",
}


def _mlx_available() -> bool:
    if sys.platform != "darwin" or platform.machine() != "arm64":
        return False
    try:
        import mlx_lm  # noqa: F401
        return True
    except ImportError:
        return False


def resolve_backend(requested: str = None) -> str:
    requested = (requested or LOCAL_LLM_BACKEND or "auto").lower()
    if requested == "auto":
        return "mlx" if _mlx_available() else "transformers"
    if requested not in DEFAULT_MODELS:
        raise ValueError(f"Unknown LOCAL_LLM_BACKEND '{requested}' (use auto, mlx or transformers)")
    return requested


class LocalHFModel:
    """Lazily-loaded local chat model. Thread-safe; loads weights on first generate()."""

    def __init__(self, model_id: str = None, backend: str = None, max_new_tokens: int = None):
        self.backend = resolve_backend(backend)
        self.model_id = model_id or LOCAL_LLM_MODEL or DEFAULT_MODELS[self.backend]
        self.max_new_tokens = int(max_new_tokens or LOCAL_LLM_MAX_NEW_TOKENS)
        self._model = None
        self._tokenizer = None
        self._device = None
        self._lock = threading.Lock()
        self.load_error: Optional[str] = None

    @property
    def name(self) -> str:
        return f"hf:{self.model_id}"

    def _load(self) -> None:
        if self._model is not None:
            return
        print(f"📥 Loading local open-weights model '{self.model_id}' ({self.backend}); first run downloads weights from Hugging Face...")
        if self.backend == "mlx":
            from mlx_lm import load
            self._model, self._tokenizer = load(self.model_id)
        else:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
            if torch.cuda.is_available():
                self._device = "cuda"
            elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
                self._device = "mps"
            else:
                self._device = "cpu"
            self._tokenizer = AutoTokenizer.from_pretrained(self.model_id)
            self._model = AutoModelForCausalLM.from_pretrained(self.model_id, torch_dtype="auto").to(self._device)
            self._model.eval()
        print(f"✅ Local model ready: {self.model_id}")

    def _chat_prompt(self, prompt: str) -> str:
        messages = [{"role": "user", "content": prompt}]
        return self._tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    def generate(self, prompt: str, max_tokens: int = None, temperature: float = 0.0) -> str:
        n_new = min(int(max_tokens or self.max_new_tokens), self.max_new_tokens)
        with self._lock:
            try:
                self._load()
            except Exception as e:
                self.load_error = str(e)
                raise RuntimeError(f"Local model '{self.model_id}' unavailable: {e}") from e

            text_prompt = self._chat_prompt(prompt)
            if self.backend == "mlx":
                from mlx_lm import generate as mlx_generate
                from mlx_lm.sample_utils import make_sampler
                out = mlx_generate(
                    self._model, self._tokenizer, prompt=text_prompt,
                    max_tokens=n_new, sampler=make_sampler(temp=float(temperature or 0.0)),
                    verbose=False,
                )
            else:
                import torch
                inputs = self._tokenizer(text_prompt, return_tensors="pt").to(self._device)
                gen_kwargs = {"max_new_tokens": n_new, "pad_token_id": self._tokenizer.eos_token_id}
                if temperature and temperature > 0:
                    gen_kwargs.update(do_sample=True, temperature=float(temperature), top_p=0.9)
                else:
                    gen_kwargs.update(do_sample=False)
                with torch.no_grad():
                    ids = self._model.generate(**inputs, **gen_kwargs)
                out = self._tokenizer.decode(ids[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)

        if not out or not out.strip():
            raise RuntimeError("Local model returned empty output")
        return out
