"""Static architectural governance for the ESMF-0 Living Observatory.

No WebGL performance, physical organism state or semantic emergence claim.
"""
import json
import unittest
from html.parser import HTMLParser
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"
ALLOWED_EVIDENCE = {"P", "A", "M", "H", "S", "R"}
ALLOWED_STAGE = {"main", "draft", "concept"}
REPO = "https://github.com/Dojo-1/ESMF-0-Mathematical-Genesis"


class _HTML(HTMLParser):
    def __init__(self):
        super().__init__()
        self.ids = set()
        self.local_assets = set()
        self.scripts = set()
        self.links = set()

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if "id" in attrs:
            if attrs["id"] in self.ids:
                raise AssertionError("duplicate HTML id: " + attrs["id"])
            self.ids.add(attrs["id"])
        for key in ("href", "src"):
            value = attrs.get(key, "")
            if value.startswith("./"):
                self.local_assets.add(value[2:])
        if tag == "script":
            self.scripts.add(attrs.get("src"))
        if tag == "a":
            self.links.add(attrs.get("href"))


class TestObservatorySite(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = json.loads((DOCS / "architecture.json").read_text())

    def test_manifest_has_explicit_schematic_authority(self):
        data = self.data
        self.assertEqual(data["schema"], "esmf-observatory/v1")
        self.assertEqual(data["kind"], "architectural-schematic")
        self.assertIn("not live", data["disclaimer"].lower())
        self.assertGreaterEqual(len(data["nodes"]), 14)
        self.assertGreaterEqual(len(data["domains"]), 4)
        self.assertGreaterEqual(len(data["hyperedges"]), 2)

    def test_every_node_has_local_type_evidence_and_verifiable_source(self):
        ids = set()
        domains = {d["id"] for d in self.data["domains"]}
        for node in self.data["nodes"]:
            self.assertNotIn(node["id"], ids)
            ids.add(node["id"])
            self.assertIn(node["domain"], domains)
            self.assertIn(node["evidence"], ALLOWED_EVIDENCE)
            self.assertIn(node["stage"], ALLOWED_STAGE)
            self.assertEqual(len(node["position"]), 3)
            self.assertGreater(node["size"], 0)
            self.assertTrue(node["source"].startswith(REPO), node["id"])
            self.assertGreater(len(node["summary"]), 15)
            self.assertGreater(len(node["detail"]), 20)
            self.assertTrue(node["label"].strip())

    def test_directed_and_higher_order_relations_resolve(self):
        ids = {n["id"] for n in self.data["nodes"]}
        for link in self.data["links"]:
            self.assertIn(link["from"], ids)
            self.assertIn(link["to"], ids)
            self.assertNotEqual(link["from"], link["to"])
            self.assertTrue(link["type"])
        for h in self.data["hyperedges"]:
            self.assertGreaterEqual(len(h["members"]), 3)
            self.assertTrue(set(h["members"]).issubset(ids))
            self.assertIn(h["evidence"], ALLOWED_EVIDENCE)
            self.assertTrue(h["source"].startswith(REPO))

    def test_measured_cards_keep_their_limits(self):
        cards = {v["id"]: v for v in self.data["findings"]}
        self.assertIn("d0", cards)
        self.assertIn("d1a", cards)
        self.assertIn("no clear", cards["d0"]["label"].lower())
        self.assertIn("not", cards["d0"]["footnote"].lower())
        self.assertIn("not", cards["d1a"]["footnote"].lower())
        self.assertTrue(all(x["evidence"] == "M" for x in cards.values()))
        self.assertTrue(all(x["source"].startswith(REPO) for x in cards.values()))

    def test_html_is_self_contained_with_accessible_fallback(self):
        parser = _HTML()
        parser.feed((DOCS / "index.html").read_text())
        for name in parser.local_assets:
            self.assertTrue((DOCS / name).exists(), name)
        for required in ("scene-viewport", "scene-fallback", "domain-controls",
                         "inspector-content", "morph-range", "node-directory"):
            self.assertIn(required, parser.ids)
        self.assertIn("./app.js", parser.scripts)
        self.assertTrue((DOCS / "fallback.svg").exists())
        self.assertTrue((DOCS / ".nojekyll").exists())

    def test_script_explicitly_declares_illustrative_not_live_mode(self):
        ui = (DOCS / "app.js").read_text()
        scene = (DOCS / "scene.js").read_text()
        self.assertIn("not live", self.data["disclaimer"].lower())
        self.assertIn("not live", ui.lower())
        self.assertIn("SCHEMATIC", (DOCS / "index.html").read_text())
        self.assertTrue("not live" in scene.lower() or "never simulates" in scene.lower())
        self.assertIn("setMorphology", scene)
        self.assertNotIn("eval(", ui)
        self.assertNotIn("eval(", scene)

    def test_recorded_replay_controls_are_explicitly_not_live(self):
        html = (DOCS / "index.html").read_text()
        ui = (DOCS / "trace-player.js").read_text()
        scene = (DOCS / "scene.js").read_text()
        workflow = (ROOT / ".github/workflows/observatory-pages.yml").read_text()
        parser = _HTML()
        parser.feed(html)
        for item in ("trace-toggle", "trace-controls", "trace-run", "trace-range",
                     "trace-back", "trace-play", "trace-next", "trace-phase",
                     "trace-snapshot", "trace-source", "trace-download",
                     "trace-payload-id"):
            self.assertIn(item, parser.ids)
        self.assertTrue((DOCS / "trace-player.js").exists())
        self.assertIn("initializeTracePlayer", (DOCS / "app.js").read_text())
        self.assertIn("conditioned", ui)
        self.assertIn("sampled", ui)
        self.assertIn("not live", ui.lower())
        self.assertIn("setReplay", scene)
        self.assertIn("clearReplay", scene)
        self.assertIn("traceQuaternionFrame", scene)
        self.assertIn("S3 -> SO(3)", scene)
        self.assertIn("python -m research.export_observatory_trace", workflow)
        self.assertIn("actions/deploy-pages@", workflow)
        self.assertIn("not randomly sampled", ui.lower())
        self.assertNotIn("eval(", ui)

    def test_g0_ah_microscope_is_recorded_not_an_anatomical_claim(self):
        self.assertIn("compartments", {n["id"] for n in self.data["nodes"]})
        node=next(n for n in self.data["nodes"] if n["id"]=="compartments")
        self.assertEqual(node["evidence"], "M")
        self.assertIn("ordinary", node["detail"].lower())
        cards={finding["id"]:finding for finding in self.data["findings"]}
        self.assertIn("ah-d0",cards)
        self.assertIn("ordinary",cards["ah-d0"]["label"].lower())
        page=(DOCS / "neurite.html").read_text()
        script=(DOCS / "neurite.js").read_text()
        parser=_HTML()
        parser.feed(page)
        for local in parser.local_assets:
            self.assertTrue((DOCS / local).exists(),local)
        for required in ("run","cell","frame","viewport","fallback",
                         "detail","remainder","hash","focus"):
            self.assertIn(required,parser.ids)
        self.assertIn("./neurite.js",parser.scripts)
        self.assertIn('href="./neurite.html"',(DOCS / "index.html").read_text())
        self.assertIn("not drawn from a stochastic generator",(
            ROOT / "research/export_g0_ah_microscope.py").read_text())
        self.assertIn("selected",script)
        self.assertIn("not a stochastic event",page.lower() + script.lower())
        self.assertNotIn("eval(",script)
        workflow=(ROOT / ".github/workflows/observatory-pages.yml").read_text()
        self.assertIn("python -m research.export_g0_ah_microscope",workflow)

    def test_merged_research_is_shown_as_main_source_not_as_validated_science(self):
        self.assertNotIn("draft", {n["stage"] for n in self.data["nodes"]})
        for node in self.data["nodes"]:
            if node["stage"] == "main":
                self.assertIn("/blob/main/", node["source"])
        self.assertEqual(next(n for n in self.data["nodes"]
                              if n["id"] == "typedmemory")["evidence"], "M")
        self.assertEqual(next(n for n in self.data["nodes"]
                              if n["id"] == "glia")["evidence"], "H")
        self.assertIn("synthetic", next(n for n in self.data["nodes"]
                                       if n["id"] == "typedmemory")["detail"].lower())


if __name__ == "__main__":
    unittest.main()
