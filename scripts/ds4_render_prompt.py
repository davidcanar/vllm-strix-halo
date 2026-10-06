# Render a DS4 chat prompt with the 0.31 deepseek_v4 tokenizer (CPU only), thinking on vs off.
import inspect, json
from vllm.tokenizers import get_tokenizer
M = "/home/davidcanar/models/DeepSeek-V4-Flash-Vision-Exp"
tok = get_tokenizer(M, tokenizer_mode="deepseek_v4", trust_remote_code=True)
print("tokenizer:", type(tok).__name__)
msgs = [{"role": "user", "content": "What is the capital of France? Answer in one word."}]
sig = inspect.signature(tok.apply_chat_template)
print("apply_chat_template params:", list(sig.parameters)[:12])
for kw in ({"thinking": False}, {"thinking": True, "reasoning_effort": "high"}, {"thinking": True}):
    try:
        out = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, **kw)
    except TypeError:
        out = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, chat_template_kwargs=kw)
    ids = tok.encode(out, add_special_tokens=False) if isinstance(out, str) else out
    text = out if isinstance(out, str) else tok.decode(out)
    print(f"\n==== {json.dumps(kw)}  ({len(ids)} tokens)")
    print(repr(text[-700:]))
