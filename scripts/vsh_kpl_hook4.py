_v0 = int(seq_lens.reshape(-1)[0])
_g = globals()
if index_kpool > 1 and _v0 >= 3000 and "layers.11." in k_cache_prefix and _g.get("_vh4_n", 0) < 6:
    _g["_vh4_n"] = _g.get("_vh4_n", 0) + 1
    _kv = kv_cache.view(torch.uint8).reshape(kv_cache.shape[0], -1)
    _scales = _kv[:, 64 * 128:64 * 128 + 256].view(torch.float32)   # [pages, 64]
    _nzp = (_scales != 0).any(dim=1).nonzero().flatten()
    _ln = "v0=%d nonzero_pages=%d first=%s | " % (_v0, _nzp.numel(), _nzp[:80].tolist())
    try:
        _ln += "dec_slot=%s dec_pos=%s " % (dec_slot.reshape(-1)[:8].tolist(), dec_pos.reshape(-1)[:8].tolist())
    except Exception as _e:
        _ln += "dec_slot n/a (%r) " % (_e,)
    try:
        _ln += "slot_mapping=%s positions=%s" % (slot_mapping[:8].tolist(), positions[:8].tolist())
    except Exception as _e:
        _ln += "sm n/a (%r)" % (_e,)
    with open("/tmp/vsh_kpl_hook4.log", "a") as _f:
        _f.write(_ln + "\n")
