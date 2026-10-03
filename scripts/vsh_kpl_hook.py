import json as _j
_nd = _j.load(open("/tmp/vsh_needles.json"))
_v0 = int(seq_lens.reshape(-1)[0])
if index_kpool > 1 and _v0 >= 3000:
    _g = globals(); _c = _g.get("_vh_c", 0) + 1; _g["_vh_c"] = _c
    if _c % 3 == 0:
        _lg = logits[0, :_v0].float()
        _order = torch.argsort(_lg, descending=True)
        _rank = torch.empty_like(_order); _rank[_order] = torch.arange(_v0, device=_order.device)
        _sel = set(pool_topk[0].tolist())
        _r = []; _ins = 0; _nm = -1e30
        for _p in _nd:
            if _p < _v0:
                _r.append(int(_rank[_p])); _nm = max(_nm, float(_lg[_p]))
                if _p in _sel: _ins += 1
        with open("/tmp/vsh_kpl_hook.log", "a") as _f:
            _f.write("%s c=%d v0=%d rows=%d best_needle_rank=%d needle_ranks=%s in_sel=%d/%d top_logit=%.2f needle_max_logit=%.2f median_logit=%.2f\n" % (
                k_cache_prefix, _c, _v0, num_rows, min(_r), _r, _ins, len(_nd),
                float(_lg.max()), _nm, float(_lg.median())))
