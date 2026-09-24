from __future__ import annotations

from .calibration import Calibration, script_bucket
from .tokenizer import HFTokenizer, Unverifiable


class LocalHFScorer:
    """Reference model on local weights: log p(ids | chat-templated prompt) at temperature 1."""

    def __init__(self, model_id: str, device: str | None = None):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
        self.model_id = model_id
        self.hf_tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForCausalLM.from_pretrained(model_id, dtype="auto").to(self.device).eval()

    def prompt_ids(self, request: dict) -> list[int]:
        out = self.hf_tokenizer.apply_chat_template(
            request["messages"], tools=request.get("tools"), add_generation_prompt=True, tokenize=True
        )
        if hasattr(out, "keys"):
            out = out["input_ids"]
        return list(out[0] if out and isinstance(out[0], list) else out)

    def logprob(self, request: dict, ids: list[int]) -> float:
        return self.logprob_ids(self.prompt_ids(request), ids)

    def logprob_ids(self, prompt: list[int], ids: list[int]) -> float:
        torch = self.torch
        x = torch.tensor([prompt + ids], device=self.device)
        with torch.no_grad():
            logits = self.model(x).logits[0, len(prompt) - 1 : -1].float()
        target = torch.tensor(ids, device=self.device)[:, None]
        return float(torch.log_softmax(logits, -1).gather(1, target).sum())

    def resolve(self, request: dict, candidates: list[list[int]]) -> list[int]:
        """Left to right, pick the candidate id the model finds most likely; rerun only after a change."""
        torch = self.torch
        prompt = self.prompt_ids(request)
        ids = [cs[0] for cs in candidates]
        pending = [i for i, cs in enumerate(candidates) if len(cs) > 1]
        while pending:
            with torch.no_grad():
                logits = self.model(torch.tensor([prompt + ids], device=self.device)).logits[0]
            for n, i in enumerate(pending):
                row = logits[len(prompt) + i - 1]
                best = max(candidates[i], key=lambda t: float(row[t]))
                if best != ids[i]:
                    ids[i], pending = best, pending[n + 1:]
                    break
            else:
                break
        return ids

    def sample(self, messages: list[dict], temperature: float, top_p: float, max_new_tokens: int) -> tuple[list[int], list[int]]:
        """One honest response: (prompt ids, generated ids without the stop token)."""
        torch = self.torch
        prompt = self.prompt_ids({"messages": messages})
        x = torch.tensor([prompt], device=self.device)
        pad = self.hf_tokenizer.pad_token_id or self.hf_tokenizer.eos_token_id
        with torch.no_grad():
            out = self.model.generate(
                x, attention_mask=torch.ones_like(x), do_sample=True, temperature=temperature,
                top_p=top_p, top_k=0, max_new_tokens=max_new_tokens, pad_token_id=pad,
            )
        eos = self.model.generation_config.eos_token_id
        stops = set(eos if isinstance(eos, list) else [eos]) | {pad}
        gen = []
        for t in out[0, len(prompt):].tolist():
            if t in stops:
                break
            gen.append(t)
        return prompt, gen


def commission(scorer: LocalHFScorer, served_model: str, prompts: list[list[dict]], calibration: Calibration,
               temperature: float = 1.0, top_p: float = 1.0, max_new_tokens: int = 512, language: str | None = None) -> None:
    """Calibrate on honest local generations with the endpoint's weights, template and sampling."""
    tok = HFTokenizer.from_pretrained(scorer.model_id)
    for messages in prompts:
        prompt, gen = scorer.sample(messages, temperature, top_p, max_new_tokens)
        data = b"".join(map(tok.token_bytes, gen))
        try:
            canon = tok.canonical(data)
        except Unverifiable:
            continue
        score = 0.0 if canon == gen else scorer.logprob_ids(prompt, gen) - scorer.logprob_ids(prompt, canon)
        calibration.add(Calibration.key(served_model, language or script_bucket(data.decode(errors="replace"))), score)
