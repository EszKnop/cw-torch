from random import uniform
import unittest
import torch
from src.cw_torch.metric import cw_normality

class TestNormalityMeasure(unittest.TestCase):

    def test_distance_for_standard_normal_samples_are_smaller_than_these_for_uniform_ones_when_using_the_same_gamma(self):
        N, D = (100, 40)
        standard_sample = torch.randn((N, D))
        uniform_sample = torch.rand((N, D))
        gamma = torch.ones((1, ))

        standard_value = cw_normality(standard_sample, gamma)
        uniform_value = cw_normality(uniform_sample, gamma)
        
        self.assertLess(standard_value, uniform_value)

if __name__ == '__main__':
    unittest.main()