import unittest
import torch
from src.cw_torch.gamma import silverman_rule_of_thumb

class TestSilvermanRuleOfThumb(unittest.TestCase):

    def test_normal_distribution_scenario_with_non_tensor_count(self):
        count, stddev = (64, torch.tensor(1.0))

        gamma_value = silverman_rule_of_thumb(stddev, count)
        
        self.assertAlmostEqual(gamma_value.item(), 0.21248091634, places=5)

    def test_normal_distribution_scenario_with_tensor_count(self):
        count, stddev = (torch.tensor(32), torch.tensor(1.0))

        gamma_value = silverman_rule_of_thumb(stddev, count)
        
        self.assertAlmostEqual(gamma_value.item(), 0.28037025, places=5)
    
    def test_not_normal_scenario_without_tensor(self):
        count, stddev = (16, 0.1)

        gamma_value = silverman_rule_of_thumb(stddev, count)
        
        self.assertAlmostEqual(gamma_value.item(), 0.00369950762, places=8)

if __name__ == '__main__':
    unittest.main()