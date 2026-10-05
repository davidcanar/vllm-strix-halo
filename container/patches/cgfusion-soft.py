import pathlib, py_compile

P = pathlib.Path("/opt/venv/lib/python3.12/site-packages/vllm/models/deepseek_v4/amd/rocm.py")
s = P.read_text()
if "vsh-cgfusion-soft" in s:
    print("already soft")
else:
    old = """        if main_weight.ndim != 2 or indexer_weight.ndim != 2:
            raise ValueError("DeepSeek V4 compressor weights must be matrices")
        if main_weight.shape[1] != indexer_weight.shape[1]:
            raise ValueError("DeepSeek V4 compressor weights must share K")"""
    new = """        if main_weight.ndim != 2 or indexer_weight.ndim != 2 or main_weight.shape[1] != indexer_weight.shape[1]:  # vsh-cgfusion-soft
            logger.warning_once(
                "DeepSeek V4 compressor GEMM fusion skipped: weights are not "
                "fusible matrices (ndim %d/%d, K %s/%s) - using the unfused path",
                main_weight.ndim, indexer_weight.ndim,
                main_weight.shape[1:], indexer_weight.shape[1:],
            )
            return False"""
    assert s.count(old) == 1, f"anchor {s.count(old)}"
    P.write_text(s.replace(old, new, 1))
    py_compile.compile(str(P), doraise=True, cfile="/tmp/_cgf.pyc")
    print("fusion soft-fail patched + syntax OK")
