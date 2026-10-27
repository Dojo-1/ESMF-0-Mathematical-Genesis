import unittest

from esmf.bundle_o import selfcheck


class TestBundleO(unittest.TestCase):
    def test_matched_flat_and_curved_controls(self):
        result = selfcheck()
        errors = result["mean_squared_transport_error"]
        self.assertTrue(result["identical_sensory_paths"])
        self.assertTrue(result["equal_matrix_work"])
        self.assertLess(errors["flat"]["flat"], 1e-24)
        self.assertLess(errors["curved"]["curved"], 1e-24)
        self.assertGreater(errors["curved"]["flat"], 1e-8)
        self.assertGreater(errors["flat"]["curved"], 1e-8)


if __name__ == "__main__":
    unittest.main()
