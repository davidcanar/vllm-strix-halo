import pathlib, py_compile

P = pathlib.Path("/opt/venv/lib/python3.12/site-packages/vllm/models/deepseek_v4/amd/rocm.py")
s = P.read_text()
if "vsh-cgdiag" in s:
    print("already instrumented")
else:
    old = """            logger.warning_once(
                "DeepSeek V4 compressor GEMM fusion skipped: weights are not "
                "fusible matrices (ndim %d/%d, K %s/%s) - using the unfused path",
                main_weight.ndim, indexer_weight.ndim,
                main_weight.shape[1:], indexer_weight.shape[1:],
            )
            return False"""
    new = """            if os.environ.get("VSH_DS4_CGDEBUG"):  # vsh-cgdiag
                try:
                    lw = compressor.fused_wkv_wgate
                    ir = indexer.compressor.fused_wkv_wgate
                    print(
                        f"[vsh-cgdiag] rank={torch.distributed.get_rank()} "
                        f"layer={getattr(compressor, 'prefix', '?')} "
                        f"main_type={type(lw).__name__} idx_type={type(ir).__name__} "
                        f"main_w={tuple(main_weight.shape)}/{main_weight.dtype} "
                        f"idx_w={tuple(indexer_weight.shape)}/{indexer_weight.dtype} "
                        f"main_out={getattr(lw, 'output_size', None)} "
                        f"main_outparts={getattr(lw, 'output_partition_sizes', None)} "
                        f"main_in={getattr(lw, 'input_size', None)} "
                        f"main_quant={type(getattr(lw, 'quant_method', None)).__name__} "
                        f"wq={getattr(lw, 'vsh_w8_q', None) is not None} "
                        f"named_params={[(n, tuple(p.shape)) for n, p in list(lw.named_parameters())[:4]]}",
                        flush=True,
                    )
                except Exception as _e:  # noqa: BLE001
                    print(f"[vsh-cgdiag] dump failed: {_e!r}", flush=True)
            logger.warning_once(
                "DeepSeek V4 compressor GEMM fusion skipped: weights are not "
                "fusible matrices (ndim %d/%d, K %s/%s) - using the unfused path",
                main_weight.ndim, indexer_weight.ndim,
                main_weight.shape[1:], indexer_weight.shape[1:],
            )
            return False"""
    assert s.count(old) == 1, f"anchor {s.count(old)}"
    s = s.replace(old, new, 1)
    if "\nimport os\n" not in s[:2000]:
        s = s.replace("import torch", "import os\nimport torch", 1)
    P.write_text(s)
    py_compile.compile(str(P), doraise=True, cfile="/tmp/_cgd.pyc")
    print("diag patched + syntax OK")
