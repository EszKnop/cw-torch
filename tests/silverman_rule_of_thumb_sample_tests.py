import unittest
import torch
from src.cw_torch.gamma import silverman_rule_of_thumb_sample

class TestSilvermanRuleOfThumbSample(unittest.TestCase):

    def test_scenario_with_zero_stddev_count(self):
        count, dim = 64, 8
        sample = torch.ones((count, dim))

        gamma_value = silverman_rule_of_thumb_sample(sample)
        
        self.assertAlmostEqual(gamma_value.item(), 0.0, places=5)
    
    def test_multivariate_scenario_with_nonzero_stddev(self):
        sample = torch.FloatTensor([[ 1.0564, -0.9815,  0.5661,  0.2148],
                [ 0.3336, -0.2038, -0.6857, -0.4036],
                [ 0.0502, -0.8962, -0.2114, -0.9635],
                [-0.1723, -0.1886, -0.2185,  1.4890],
                [-2.4784,  1.0248,  1.2122, -1.5725],
                [ 1.4533,  0.8522,  0.9728,  0.6231],
                [-0.4511, -0.0140, -1.4338,  0.5991],
                [-0.7231, -1.5806,  0.4648, -0.1251]])

        gamma_value = silverman_rule_of_thumb_sample(sample)
        
        self.assertAlmostEqual(gamma_value.item(), 0.44628402056421074, places=5)


if __name__ == '__main__':
    unittest.main()