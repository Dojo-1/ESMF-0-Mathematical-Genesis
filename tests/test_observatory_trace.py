"""Trace export is a bounded software witness, never a biological validation."""
import hashlib
import json
import math
import tempfile
import unittest
from pathlib import Path

from esmf.glial_quaternion_ag import GlialQuaternionEcology
from research.export_observatory_trace import (
    SCHEMA, PRELUDE, export_trace, fingerprint, snapshot, write_trace,
)


class TestObservatoryTrace(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.trace = export_trace("a" * 40)

    def test_source_provenance_and_checksum(self):
        data = self.trace
        self.assertEqual(data["schema"], SCHEMA)
        self.assertEqual(data["source_commit"], "a" * 40)
        self.assertEqual(len(data["runs"]), 2)
        payload = dict(data)
        checksum = payload.pop("payload_sha256")
        self.assertEqual(fingerprint(payload), checksum)
        self.assertIn("ONLY", data["authority"])
        self.assertTrue(any("not stochastic" in s for s in
                            data["limitations"]))

    def test_live_state_dimensions_and_proxy_ledgers(self):
        for run in self.trace["runs"]:
            for frame in run["frames"]:
                snap = frame["state"]
                self.assertEqual(len(snap["cells"]), 2)
                self.assertEqual(snap["ledger"]["combined_energy_residual"], 0)
                self.assertEqual(snap["ledger"]["structural_material_residual"], 0)
                self.assertEqual(fingerprint(snap), frame["state_sha256"])
                self.assertTrue(all(c["atp"] >= 0 and c["material"] >= 0
                                    for c in snap["cells"]))
                for c in snap["cells"]:
                    self.assertEqual(len(c["soma"]), 3)
                    self.assertEqual(c["real_carrier_dimension"],
                                     8 + 4 * len(c["path"]))
                    self.assertEqual(len(c["tip_quaternion"]), 4)
                    self.assertTrue(math.isfinite(c["transport_defect"]))
                self.assertGreaterEqual(snap["external_inputs"]["nutrient"], 0)
                self.assertGreaterEqual(snap["external_inputs"]["material"], 0)
                self.assertEqual(snap["glia"]["tripartite"], bool(
                    snap["glia"]["tripartite"]))

    def test_conditioned_events_are_labeled_not_passed_as_sampled(self):
        demonstration, natural = self.trace["runs"]
        self.assertEqual(demonstration["id"], "conditioned")
        self.assertEqual(demonstration["conditioned_event_count"], len(PRELUDE))
        self.assertEqual(
            [x["event"] for x in demonstration["frames"][1:len(PRELUDE) + 1]],
            list(PRELUDE),
        )
        self.assertTrue(all(
            f["phase"] == "conditioned" and f["time"] is None
            for f in demonstration["frames"][1:len(PRELUDE) + 1]
        ))
        self.assertGreater(demonstration["frames"][1]["selected_rate"], 0)
        self.assertTrue(any(
            f["state"]["glia"]["tripartite"]
            for f in demonstration["frames"]
        ))
        self.assertTrue(any(
            len(c["path"]) > 0
            for f in demonstration["frames"]
            for c in f["state"]["cells"]
        ))
        self.assertEqual(natural["conditioned_event_count"], 0)
        self.assertTrue(all(f["phase"] != "conditioned"
                            for f in natural["frames"]))
        self.assertEqual(natural["frames"][0]["event"], "initial")

    def test_sampled_situated_process_preserves_time_and_hazard(self):
        for run in self.trace["runs"]:
            times = [f["time"] for f in run["frames"]
                     if f["phase"] == "sampled"]
            self.assertEqual(times, sorted(times))
            self.assertEqual(len(set(times)), len(times))
            self.assertTrue(all(0 < t < run["horizon"] for t in times))
            self.assertEqual(run["sampled_event_count"], len(times))
            for frame in run["frames"]:
                if frame["phase"] in ("conditioned", "sampled"):
                    self.assertGreater(frame["selected_rate"], 0)
                    self.assertGreaterEqual(
                        frame["outgoing_rate_sum"], frame["selected_rate"])
                if frame["typed_remainder"] is not None:
                    self.assertTrue(
                        frame["typed_remainder"]["type"].startswith("[R]_")
                    )

    def test_export_is_deterministic_and_json_roundtrips(self):
        model = GlialQuaternionEcology()
        self.assertEqual(
            fingerprint(snapshot(model, model.seed())),
            self.trace["runs"][0]["frames"][0]["state_sha256"],
        )
        self.assertEqual(export_trace("a" * 40), self.trace)
        with tempfile.TemporaryDirectory() as d:
            path = Path(d, "trace.json")
            write_trace(path, source_sha="a" * 40)
            loaded = json.loads(path.read_text())
            self.assertEqual(loaded, self.trace)
            self.assertEqual(hashlib.sha256(
                json.dumps(loaded["runs"], sort_keys=True).encode()
            ).digest_size, 32)

    def test_source_sha_cannot_be_empty_or_unbounded(self):
        with self.assertRaises(ValueError):
            export_trace("")
        with self.assertRaises(ValueError):
            export_trace("x" * 100)


if __name__ == "__main__":
    unittest.main()
