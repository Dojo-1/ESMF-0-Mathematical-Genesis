"""G0-AI: fixed future horizon, label causality, and strong controls."""
import math
import unittest
from research.g0_ai_probe import (
    STEPS, BLOCK, SEEDS, FUTURE_RESERVED, generate, run_seed, LaggedTable,
)


class TestG0AIProbe(unittest.TestCase):
    def test_changed_world_hidden_from_structural_circuit(self):
        stream=generate(SEEDS[0])
        self.assertEqual(len(stream),STEPS)
        self.assertEqual([stream[t][3] for t in (0,BLOCK,2*BLOCK)],[0,1,0])
        self.assertTrue(all(isinstance(row[2],bool) and len(row[1])==4
                            for row in stream))
        self.assertEqual(stream,generate(SEEDS[0]))
        self.assertNotEqual(stream,generate(SEEDS[1]))
        self.assertEqual(len(FUTURE_RESERVED),32)
        self.assertTrue(set(SEEDS).isdisjoint(FUTURE_RESERVED))

    def test_oracle_feature_table_sees_only_available_history(self):
        table=LaggedTable()
        self.assertEqual(table.predict([]),.5)
        table.observe(True)
        self.assertEqual(table.predict([(1,-1,1,1)]),.5)
        table.observe(False)
        self.assertEqual(table.predict([(1,-1,1,1),(-1,1,1,1)]),.5)
        table.observe(True)
        self.assertGreater(table.predict([(1,-1,1,1),(-1,1,1,1)]),.5)

    def test_prequential_test_is_reproducible_and_separates_costs(self):
        result=run_seed(SEEDS[0])
        self.assertEqual(result['initial_units'],20)
        self.assertTrue(result['all_ledgers_closed'])
        self.assertTrue(result['model_label_used_only_by_external_readout'])
        self.assertEqual(len(result['cost']),3)
        self.assertEqual(result['world_events'],720)
        for arm in result['scores']:
            item=result['scores'][arm]
            self.assertEqual(len(item['blocks']),3)
            self.assertTrue(0 <= item['mean'] <=1 and math.isfinite(item['mean']))
        # No claim of a benefit is baked into the tests.
        self.assertEqual(result['scores'],run_seed(SEEDS[0])['scores'])
        for arm in ('adaptive','fixed','plastic_no_growth','shuffled'):
            self.assertEqual(result['cost'][-1][arm]['ledger'],(0,0))
        self.assertEqual(result['active_morphology']['fixed'],0)
        self.assertEqual(result['active_morphology']['plastic_no_growth'],0)

    def test_unregistered_seed_rejected(self):
        for seed in (SEEDS[0]-1, SEEDS[-1]+1, FUTURE_RESERVED[0]):
            with self.assertRaises(ValueError):run_seed(seed)


if __name__=='__main__':unittest.main()
