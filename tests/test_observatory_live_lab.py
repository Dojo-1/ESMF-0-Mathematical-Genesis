"""Contracts for the browser-executed G0-AI laboratory.

These are static integrity checks, not browser/WebGL benchmarking.
"""
import unittest
from pathlib import Path
from html.parser import HTMLParser

ROOT=Path(__file__).resolve().parents[1]
DOCS=ROOT/"docs"

class Page(HTMLParser):
    def __init__(self):
        super().__init__();self.ids=set();self.assets=set();self.scripts=set()
    def handle_starttag(self,tag,attrs):
        a=dict(attrs)
        if "id" in a:
            if a["id"] in self.ids:
                raise AssertionError("duplicate DOM identifier "+a["id"])
            self.ids.add(a["id"])
        for key in ("src","href"):
            v=a.get(key,"")
            if v.startswith("./"):
                self.assets.add(v[2:])
        if tag=="script" and a.get("src"):
            self.scripts.add(a["src"])

class LiveLabContract(unittest.TestCase):
    def test_runtime_snapshot_matches_real_python_model(self):
        self.assertEqual(
            (DOCS/"runtime/telos_substrate_ai.py").read_bytes(),
            (ROOT/"esmf/telos_substrate_ai.py").read_bytes(),
            "Update the browser-runtime snapshot together with the research source."
        )
    def test_live_page_has_complete_accessible_control_surface(self):
        parser=Page();parser.feed((DOCS/"live.html").read_text())
        for name in parser.assets:
            self.assertTrue((DOCS/name).is_file(),name)
        for name in ("viewport","fallback","stimulus","run","auto","reset",
                     "runtime-status","events","site","site-state",
                     "math-tab","language-tab","prediction","energy","ledger"):
            self.assertIn(name,parser.ids)
        self.assertIn("./live.js",parser.scripts)
    def test_runtime_is_actual_python_not_decorative_mock(self):
        js=(DOCS/"live.js").read_text()
        html=(DOCS/"live.html").read_text()
        self.assertIn("loadPyodide",js)
        self.assertIn("_esmf.predict",js)
        self.assertIn("_esmf.observe",js)
        self.assertIn("_esmf.integrity",js)
        self.assertIn("runtime/telos_substrate_ai.py",js)
        self.assertIn("UnrealBloomPass",js)
        self.assertIn("not held-out evaluation",html.lower())
        self.assertIn("not a live language model",html.lower())
        self.assertNotIn("eval(",js)
        self.assertIn('href="./live.html"',(DOCS/"index.html").read_text())
        self.assertIn("node --check docs/live.js",(
            ROOT/".github/workflows/observatory-pages.yml").read_text())
    def test_no_unclaimed_model_advance(self):
        html=(DOCS/"live.html").read_text()
        self.assertIn("0.2696",html)
        self.assertIn("0.2614",html)
        self.assertIn("no semantic",html.lower())
if __name__=="__main__":
    unittest.main()
