"""Build index.html from index.template.html by inlining core.js and twin.json (the page is self-contained).

    python3 receipts/build_page.py
"""
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

core = open(os.path.join(ROOT, "core.js"), encoding="utf-8").read()
twin = json.load(open(os.path.join(HERE, "twin.json"), encoding="utf-8"))
tpl = open(os.path.join(HERE, "index.template.html"), encoding="utf-8").read()
assert "/*__CORE__*/" in tpl and "/*__TWIN__*/" in tpl
html = tpl.replace("/*__CORE__*/", core).replace("/*__TWIN__*/", json.dumps(twin, separators=(",", ":")))
assert "</script>" not in core  # inlined script must not close the tag early
open(os.path.join(ROOT, "index.html"), "w", encoding="utf-8").write(html)
print("wrote index.html", len(html.encode("utf-8")), "bytes")
