"""Startup kernel warmup for the vllm-strix-halo GLM server. Launched as a
transient unit by vsh-cluster-restart.sh once the API answers; fires dummy
requests to trigger & cache the JIT compiles so the user's FIRST real request
doesn't eat the latency:
  1) a tiny request  -> vLLM Triton kernels (AWQ dequant GEMM, MLA sparse
     attention, indexer) + decode path + MTP drafter kernels
  2) one VSH_WARMUP_CTX-sized request -> long-prefill kernels through the
     indexer buckets
Best-effort: never fails the service. VSH_WARMUP=0 disables.
"""
import json, os, time, urllib.request

_PORT = os.environ.get("VSH_WARMUP_PORT", "1235")
_MODEL = os.environ.get("VSH_WARMUP_MODEL", "glm-5.3-flash")
URL = f"http://127.0.0.1:{_PORT}/v1/chat/completions"
MODELS = f"http://127.0.0.1:{_PORT}/v1/models"
WARMUP_CTX = int(os.environ.get("VSH_WARMUP_CTX", "2048"))


def log(m):
    print(f"[vsh-warmup] {m}", flush=True)


def wait_ready(timeout_s=1800):
    t0 = time.time()
    while time.time() - t0 < timeout_s:
        try:
            urllib.request.urlopen(MODELS, timeout=3).read()
            return True
        except Exception:
            time.sleep(3)
    return False


def prompt_of(approx_tokens):
    n = max(1, approx_tokens // 14)
    return "\n".join(
        f"Line {i}: id={i} value={(i * 7919) % 100003} tag={chr(65 + i % 6)}."
        for i in range(n))


def fire(prompt, max_tokens, timeout_s):
    body = json.dumps({"model": _MODEL,
                       "messages": [{"role": "user", "content": prompt}],
                       "max_tokens": max_tokens, "temperature": 0.0}).encode()
    req = urllib.request.Request(URL, data=body,
                                 headers={"Content-Type": "application/json"})
    o = json.loads(urllib.request.urlopen(req, timeout=timeout_s).read())
    return o.get("usage", {}).get("prompt_tokens", 0)


def main():
    if os.environ.get("VSH_WARMUP", "1") == "0":
        log("disabled (VSH_WARMUP=0)"); return
    if not wait_ready():
        log("server never became ready; skipping warmup"); return
    t0 = time.time()
    try:
        log("warming vLLM/decode kernels (tiny request)...")
        fire("Say ACK.", 12, 300)
    except Exception as e:
        log(f"tiny warmup skipped: {str(e)[:60]}")
    if WARMUP_CTX > 0:
        try:
            log(f"warming prefill/indexer kernels up to ~{WARMUP_CTX} ctx...")
            pt = fire(prompt_of(WARMUP_CTX) + "\nReply: ACK", 4, 2400)
            log(f"warmed with {pt} prompt tokens")
        except Exception as e:
            log(f"prefill warmup skipped: {str(e)[:60]}")
    log(f"warmup complete in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
