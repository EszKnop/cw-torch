from random import uniform
import unittest
import torch
from src.cw_torch.metric import cw_normality_no_scale

class TestNormalityMeasureNoScale(unittest.TestCase):

    def test_zero_vector_and_ones_vector(self):
        N, D = (2, 32)
        gamma = torch.tensor(0.5)
        sample = torch.zeros((N, D))
        sample[1, :] = torch.ones((1, D))

        normality_distance = cw_normality_no_scale(sample, gamma)
        
        self.assertAlmostEqual(normality_distance.item(), 0.20768175975, places=4)
   
    def test_distance_for_standard_normal_samples_are_smaller_than_these_for_uniform_ones_when_using_the_same_gamma(self):
        N, D = (100, 40)
        standard_sample = torch.randn((N, D))
        uniform_sample = torch.rand((N, D))
        gamma = torch.tensor(1.0)

        standard_value = cw_normality_no_scale(standard_sample, gamma)
        uniform_value = cw_normality_no_scale(uniform_sample, gamma)
        
        self.assertLess(standard_value, uniform_value)
        self.assertLess(standard_value, uniform_value)

    def test_result_is_zero_dim_tensor(self):
        N, D = (32, 32)
        gamma = torch.tensor(1.1)
        sample = torch.randn((N, D))

        normality_distance = cw_normality_no_scale(sample, gamma)
        
        self.assertEqual(normality_distance.ndim, 0)

    def test_behavior_for_sample_containing_only_zeros(self):
        N, D = (8, 32)
        gamma = torch.tensor(1.0)
        sample = torch.zeros((N, D))

        normality_distance = cw_normality_no_scale(sample, gamma)
        
        self.assertAlmostEqual(normality_distance.item(), 0.07411361933, places=5)

if __name__ == '__main__':
    unittest.main()