import unittest
from esmf.controls import paired

class TestControls(unittest.TestCase):
    def test_reproducibility(self):
        self.assertEqual(paired(4, 80), paired(4, 80))
    def test_matched_final_edge_count(self):
        row = paired(4, 80)
        self.assertEqual(row["fixed_matched"]["edges"], row["scripted_growth"]["edges"])
        self.assertEqual(row["fixed_empty"]["edges"], 0)

if __name__ == "__main__":
    unittest.main()
