import torch
from torch.nn import functional as F

class CombinationLoss(torch.nn.Module):
    def __init__(self, n_all_AUs, max_intensity, reduction='mean'):
        super(CombinationLoss, self).__init__()
        self.n_all_AUs = n_all_AUs
        self.max_intensity = max_intensity
        self.reduction = reduction

    def forward(self, outputs, labels, reg_weights, class_weights):
        outputs_regression = outputs[:, :self.n_all_AUs] # Regression estimations represented by the first n_all_AUs (64) entries
        outputs_classification = outputs[:, self.n_all_AUs:].reshape(-1, self.n_all_AUs, self.max_intensity) # Classification estimations represented by the other n_all_AUs*max_intensity (64*5) entries

        mask = labels != -1 # AUs without labels represented by a placefolder of -1

        # MSE loss for regression
        regression_loss_mse = F.mse_loss(outputs_regression, labels, reduction='none')
        regression_loss_mse_masked = regression_loss_mse * mask.float()
        reg_weights_masked = reg_weights * mask.float()
        weighted_regression_loss_mse = (regression_loss_mse_masked * reg_weights_masked).sum(1)# / reg_weights.sum(1)

        # Cosine similarity loss for regression
        cos_sim = F.cosine_similarity(outputs_regression * mask.float(), labels * mask.float(), dim=1)
        regression_loss_cos = 1 - cos_sim

        # Convert the raw intensity labels to labels of multiple binary classifications
        labels_classification = (labels.unsqueeze(-1) >= torch.arange(1, self.max_intensity + 1).to(labels.device)).float()
        mask_classification = mask.unsqueeze(-1).expand(-1, -1, self.max_intensity)

        # Cropp-entropy loss for classifications
        classification_loss = F.binary_cross_entropy_with_logits(outputs_classification, labels_classification, reduction='none')
        classification_loss_masked = classification_loss * mask_classification.float()
        class_weights_masked = class_weights * mask_classification.float()
        weighted_classification_loss = (classification_loss_masked * class_weights_masked).sum([1, 2])# / class_weights.sum([1, 2])

        # Reduction of the loss
        if self.reduction == 'mean':
            weighted_regression_loss_mse = weighted_regression_loss_mse.mean()
            regression_loss_cos = regression_loss_cos.mean()
            weighted_classification_loss = weighted_classification_loss.mean()
        elif self.reduction == 'sum':
            weighted_regression_loss_mse = weighted_regression_loss_mse.sum()
            regression_loss_cos = regression_loss_cos.sum()
            weighted_classification_loss = weighted_classification_loss.sum()
        elif self.reduction != 'none':
            raise ValueError(f'Unsupported reduction: {self.reduction}')

        return weighted_regression_loss_mse, regression_loss_cos, weighted_classification_loss