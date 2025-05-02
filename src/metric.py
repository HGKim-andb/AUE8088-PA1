from torchmetrics import Metric
import torch

# [TODO] Implement this!
class MyF1Score(Metric):
    def __init__(self, num_classes=200):
        super().__init__()
        self.num_classes = num_classes
        
        # 클래스별 카운터 초기화
        self.add_state("tp", default=torch.zeros(num_classes), dist_reduce_fx="sum")
        self.add_state("fp", default=torch.zeros(num_classes), dist_reduce_fx="sum")
        self.add_state("fn", default=torch.zeros(num_classes), dist_reduce_fx="sum")
    
    def update(self, preds, target):
        # 예측값에서 최대 점수 클래스 선택 (벡터화된 연산)
        pred_classes = torch.argmax(preds, dim=1)
        
        # 각 클래스별로 TP, FP, FN 계산 (벡터화)
        for c in range(self.num_classes):
            # 마스크 생성
            target_mask = (target == c)
            pred_mask = (pred_classes == c)
            
            # True Positives
            self.tp[c] += (target_mask & pred_mask).sum()
            
            # False Positives
            self.fp[c] += (~target_mask & pred_mask).sum()
            
            # False Negatives
            self.fn[c] += (target_mask & ~pred_mask).sum()
    
    def compute(self):
        # 0으로 나누기 방지를 위한 작은 값
        eps = 1e-10
        
        # 클래스별 정밀도, 재현율 계산
        precision = self.tp / (self.tp + self.fp + eps)
        recall = self.tp / (self.tp + self.fn + eps)
        
        # F1 점수 계산
        f1 = 2 * (precision * recall) / (precision + recall + eps)
        
        # 클래스별 점수 평균
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
        assert predicted_classes.shape == target.shape, "Predictions and targets must have the same shape"

        # [TODO] Cound the number of correct prediction
        correct = (predicted_classes == target).sum()

        # Accumulate to self.correct
        self.correct += correct

        # Count the number of elements in target
        self.total += target.numel()

    def compute(self):
        return self.correct.float() / self.total.float()

