import ast
from pathlib import Path
p = Path("/opt/venv/lib/python3.12/site-packages/vllm/models/glm5next/amd/sparse_indexer.py")
s = p.read_text()
MARK = "# [vsh-kpl-hook]"
ANCHOR = "        # [vsh-kpl-diag]\n"
HOOK = '''        # [vsh-kpl-hook] debug: exec /tmp/vsh_kpl_hook.py (if present) with the decode locals
        import os as _vh_os
        if _vh_os.path.exists("/tmp/vsh_kpl_hook.py"):
            try:
                exec(open("/tmp/vsh_kpl_hook.py").read(), globals(), dict(locals()))
            except Exception as _vh_e:
                with open("/tmp/vsh_kpl_hook.err", "a") as _f:
                    _f.write(repr(_vh_e) + chr(10))
'''
if MARK not in s:
    assert s.count(ANCHOR) == 1
    s = s.replace(ANCHOR, HOOK + ANCHOR)
ast.parse(s); p.write_text(s); print("hook installed")
