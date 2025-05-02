from torchmetrics import Metric
import torch

# [TODO] Implement this!
class MyF1Score(Metric):
    def __init__(self, num_classes=200):
        super().__init__()
        self.num_classes = num_classes
        
        self.add_state("tp", default=torch.zeros(num_classes), dist_reduce_fx="sum")
        self.add_state("fp", default=torch.zeros(num_classes), dist_reduce_fx="sum")
        self.add_state("fn", default=torch.zeros(num_classes), dist_reduce_fx="sum")
    
    def update(self, preds, target):
        pred_classes = torch.argmax(preds, dim=1)
        
        for c in range(self.num_classes):
            target_mask = (target == c)
            pred_mask = (pred_classes == c)
            
            self.tp[c] += (target_mask & pred_mask).sum()
            
            self.fp[c] += (~target_mask & pred_mask).sum()
            
            self.fn[c] += (target_mask & ~pred_mask).sum()
    
    def compute(self):
        eps = 1e-10
        
        precision = self.tp / (self.tp + self.fp + eps)
        recall = self.tp / (self.tp + self.fn + eps)
        
        f1 = 2 * (precision * recall) / (precision + recall + eps)
        
        return torch.mean(f1)

class MyAccuracy(Metric):
    def __init__(self):
        super().__init__()
        self.add_state('total', default=torch.tensor(0), dist_reduce_fx='sum')
        self.add_state('correct', default=torch.tensor(0), dist_reduce_fx='sum')

    def update(self, preds, target):
        # [TODO] The preds (B x C tensor), so take argmax to get index with highest confidence
        predicted_classes = torch.argmax(preds, dim=1)

        # [TODO] check if preds and target have equal shape
        assert predicted_classes.shape == target.shape

        # [TODO] Cound the number of correct prediction
        correct = (predicted_classes == target).sum()

        # Accumulate to self.correct
        self.correct += correct

        # Count the number of elements in target
        self.total += target.numel()

    def compute(self):
        return self.correct.float() / self.total.float()

