import unittest
import torch
from Metrics.Metrics import cosine_similarity_matrix, tensor_text_to_video_metrics

class TestMetrics(unittest.TestCase):
    def test_cosine_similarity_matrix(self):
        v1 = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
        v2 = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
        sim = cosine_similarity_matrix(v1, v2)
        self.assertTrue(torch.equal(sim, torch.eye(3)))
    def test_tensor_text_to_video_metrics(self):
        sim = torch.eye(3).unsqueeze(0)
        metrics = tensor_text_to_video_metrics(sim, [1, 2, 3])
        self.assertEqual(metrics, {'R1': 100.0, 'R2': 100.0, 'R3': 100.0, 'MedianR': 1.0, 'MeanR': 1.0, 'Std_Rank': 0.0, 'MR': 1.0})