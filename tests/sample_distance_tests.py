import unittest
import torch
from src.cw_torch.metric import cw

class TestSampleDistance(unittest.TestCase):

    def test_sample_from_itself_is_zero(self):
        N, D = (2, 32)
        sample = torch.randn((N, D))
        gamma = 1.0

        distance = cw(sample, sample, gamma)
        
        self.assertAlmostEqual(distance.item(), 0.0, places=5)
        
    def test_symmetry(self):
        N, D = (128, 16)
        first_sample = torch.randn((N, D))
        second_sample = torch.randn((N, D))
        gamma = 0.3

        first_distance = cw(first_sample, second_sample, gamma)
        second_distance = cw(second_sample, first_sample, gamma)
        
        self.assertAlmostEqual(first_distance.item(), second_distance.item(), places=5)
        
    def test_triangle_condition(self):
        N, D = (64, 8)
        first_sample = torch.randn((N, D))
        second_sample = torch.randn((N, D))
        third_sample = torch.randn((N, D))
        gamma = 0.2

        first_distance = cw(first_sample, second_sample, gamma)
        second_distance = cw(second_sample, third_sample, gamma)
        third_distance = cw(first_sample, third_sample, gamma)
        
        self.assertLessEqual(third_distance.item(), first_distance.item() + second_distance.item())
   
    def test_simple_scenario(self):
        gamma = torch.tensor(0.25)
        first_sample = torch.Tensor([[ 0.8552, -0.4313,  0.6838, -1.3297],
            [-0.9460,  0.0400, -0.3971, -0.6276],
            [ 0.5923, -0.5162,  0.3167, -1.9416],
            [ 0.0510, -2.1722,  0.0046, -0.8541],
            [-0.8097, -1.6507, -1.7508, -1.0506],
            [ 0.5821, -1.0767,  1.4038,  0.2506],
            [-0.0436, -0.5890,  0.0834, -0.5067],
            [-1.5162, -0.4734, -1.3553, -0.4061]])
        second_sample = torch.Tensor([[ 0.0627,  0.7577, -1.4780,  1.0068],
            [ 0.3011,  0.6528, -0.4111, -0.3062],
            [-0.7347,  1.0980,  0.4572, -2.5876],
            [-0.9036,  0.2979, -0.2190,  0.0227],
            [-0.8610, -1.4610,  1.9940,  1.1780],
            [-0.3901, -0.6019, -0.3916, -0.4636],
            [ 0.9056, -0.5749,  0.1444, -1.5109],
            [ 1.6627,  0.8215, -0.5497, -1.7971]])

        result = cw(first_sample, second_sample, gamma)

        self.assertAlmostEqual(result.item(), 0.0778394117951393, places=5)
    
    def test_simple_scenario_2(self):
        first_sample = torch.Tensor([[ 0.7877,  0.5314, -0.4811,  0.8969,  0.2553,  0.1176],
        [ 0.0945,  1.2884, -2.1206,  1.4172, -1.0934, -1.0650],
        [ 0.2653,  0.4681, -0.5184, -1.1459, -0.9426, -0.7228],
        [ 0.8244, -0.3308,  0.2718, -2.0928,  1.4914,  1.0128]])

        second_sample = torch.Tensor([[ 0.1651, -0.2732,  1.7607, -0.0764, -1.1410,  0.8466],
        [ 0.3135, -0.3145, -1.0066, -0.1305,  1.8563,  1.0151],
        [ 1.0817,  0.5058,  0.1595,  1.1838,  0.0043,  0.2077],
        [-0.1303,  1.4732, -3.2696, -1.2283,  0.0055, -2.4148]])

        gamma = 0.123
        
        result = cw(first_sample, second_sample, gamma)

        self.assertAlmostEqual(result.item(), 0.21837042272090912, places=5)

    def test_two_normal_distribution_are_closer_than_uniform(self):
        N, D = 128, 32
        first_normal_sample = torch.randn((N, D))
        second_normal_sample = torch.randn((N, D))
        uniform_sample = torch.rand((N, D))
        gamma = 0.0259
        
        normal_distance = cw(first_normal_sample, second_normal_sample, gamma)
        uniform_to_normal = cw(first_normal_sample, uniform_sample, gamma)

        self.assertLessEqual(normal_distance, uniform_to_normal)


if __name__ == '__main__':
    unittest.main()