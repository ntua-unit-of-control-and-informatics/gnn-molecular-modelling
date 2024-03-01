import torch.nn as nn
import torch

class LabelSmoothingBCEWithLogitsLoss(nn.Module):
    def __init__(self,
                 smoothing=[0.0, 0.0],
                 weight=None,
                 size_average=None,
                 reduce=None,
                 reduction='mean',
                 pos_weight=None,):
        
        super(LabelSmoothingBCEWithLogitsLoss, self).__init__()
        
        self.smoothing = smoothing
        self.bce = nn.BCEWithLogitsLoss(weight, size_average, reduce, reduction, pos_weight)
        

    def forward(self, outputs, targets):
        
        label_smoothing = [torch.tensor(self.smoothing[0]).float(), torch.tensor(1-self.smoothing[1]).float()]
        smoothed_labels = torch.where(targets < 0.5, label_smoothing[0], label_smoothing[1])
        loss = self.bce(outputs, smoothed_labels)
        
        return loss
