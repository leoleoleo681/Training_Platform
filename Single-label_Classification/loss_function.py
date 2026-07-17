import torch.nn as nn
import torch

"""
模型所用所有LOSS集合
"""

# Asymmetric Loss For Multi-Label Classification
# https://openaccess.thecvf.com/content/ICCV2021/papers/Ridnik_Asymmetric_Loss_for_Multi-Label_Classification_ICCV_2021_paper.pdf

class AsymmetricLoss(nn.Module):
    def __init__(self, gamma_neg=2, gamma_pos=2, clip=0.05, eps=1e-8, disable_torch_grad_focal_loss=True,
                    label_distribution = None, distribution_gamma=0.5, alpha=2
        ):
        super(AsymmetricLoss, self).__init__()

        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.disable_torch_grad_focal_loss = disable_torch_grad_focal_loss
        self.eps = eps
        # label distribution
        self.label_distribution = label_distribution # tensor shape:[num_labels]
        # gamma: It is highly correlated with the distribution, it is recommended to experiment
        self.distribution_gamma = distribution_gamma 
        self.alpha = alpha

    def forward(self, x, y):
        """"
        Parameters
        ----------
        x: input logits
        y: targets (multi-label binarized vector)
        """

        # Calculating Probabilities
        x_sigmoid = torch.sigmoid(x)
        xs_pos = x_sigmoid
        xs_neg = 1 - x_sigmoid

        # Asymmetric Clipping
        if self.clip is not None and self.clip > 0:
            xs_neg = (xs_neg + self.clip).clamp(max=1) # check

        # Basic CE calculation
        los_pos = y * torch.log(xs_pos.clamp(min=self.eps))
        los_neg = (1 - y) * torch.log(xs_neg.clamp(min=self.eps))
        loss = los_pos + los_neg

        # Asymmetric Focusing
        if self.gamma_neg > 0 or self.gamma_pos > 0:
            pt0 = xs_pos * y
            pt1 = xs_neg * (1 - y)  # pt = p if t > 0 else 1-p
            pt = pt0 + pt1
            one_sided_gamma = self.gamma_pos * y + self.gamma_neg * (1 - y)
            one_sided_w = torch.pow(1 - pt, one_sided_gamma)
            if self.disable_torch_grad_focal_loss:
                one_sided_w = one_sided_w.detach()
            loss *= one_sided_w

        # label distribution
        if self.label_distribution is not None:
            label_distribution = self.label_distribution.to(device=x.device, dtype=x.dtype)
            label_total = torch.sum(label_distribution)
            if label_total <= 0:
                raise ValueError("label_distribution总数必须大于0")
            label_other = label_total - label_distribution
            alpha = torch.pow(label_other/label_total, self.distribution_gamma)
            loss *= alpha
        else:
            alpha = self.alpha
            if torch.is_tensor(alpha):
                alpha = alpha.to(device=x.device, dtype=x.dtype)
            loss *= alpha

        return -loss.sum()


class AsymmetricLossOptimized(nn.Module):
    ''' Notice - optimized version, minimizes memory allocation and gpu uploading,
    favors inplace operations'''

    def __init__(self, gamma_neg=4, gamma_pos=1, clip=0.05, eps=1e-8, disable_torch_grad_focal_loss=False):
        super(AsymmetricLossOptimized, self).__init__()

        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.disable_torch_grad_focal_loss = disable_torch_grad_focal_loss
        self.eps = eps

        # prevent memory allocation and gpu uploading every iteration, and encourages inplace operations
        self.targets = self.anti_targets = self.xs_pos = self.xs_neg = self.asymmetric_w = self.loss = None

    def forward(self, x, y):
        """"
        Parameters
        ----------
        x: input logits
        y: targets (multi-label binarized vector)
        """

        self.targets = y
        self.anti_targets = 1 - y

        # Calculating Probabilities
        self.xs_pos = torch.sigmoid(x)
        self.xs_neg = 1.0 - self.xs_pos

        # Asymmetric Clipping
        if self.clip is not None and self.clip > 0:
            self.xs_neg.add_(self.clip).clamp_(max=1)

        # Basic CE calculation
        self.loss = self.targets * torch.log(self.xs_pos.clamp(min=self.eps))
        self.loss.add_(self.anti_targets * torch.log(self.xs_neg.clamp(min=self.eps)))

        # Asymmetric Focusing
        if self.gamma_neg > 0 or self.gamma_pos > 0:
            self.xs_pos = self.xs_pos * self.targets
            self.xs_neg = self.xs_neg * self.anti_targets
            self.asymmetric_w = torch.pow(1 - self.xs_pos - self.xs_neg,
                                          self.gamma_pos * self.targets + self.gamma_neg * self.anti_targets)
            if self.disable_torch_grad_focal_loss:
                self.asymmetric_w = self.asymmetric_w.detach()
            self.loss *= self.asymmetric_w

        return -self.loss.sum()


class FocalLoss(nn.Module):
    def __init__(self, gamma=None, clip=0.05, eps=1e-8, disable_torch_grad_focal_loss=True,
                    label_distribution = None, distribution_gamma=None, alpha=None
        ):
        super(FocalLoss, self).__init__()

        self.clip = clip
        self.disable_torch_grad_focal_loss = disable_torch_grad_focal_loss
        self.eps = eps

        # label distribution
        self.label_distribution = label_distribution # tensor shape:[num_labels]
        # distribution gamma: It is highly correlated with the distribution, it is recommended to experiment
        self.distribution_gamma = distribution_gamma 
        # gamma 
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, x, y):
        """"
        Parameters
        ----------
        x: input logits
        y: targets (multi-class vector)
        """
        if y.dim() == 1:
            y = torch.nn.functional.one_hot(
                y.long(),
                num_classes=x.size(-1),
            ).to(dtype=x.dtype, device=x.device)
        # Calculating Probabilities
        x_sotfmax = torch.softmax(x,dim=-1)
        # Basic CE calculation
        loss = y * torch.log(x_sotfmax.clamp(min=self.eps))

        # Focal gamma
        if self.gamma > 0:
            pt = x_sotfmax * y
            one_sided_gamma = self.gamma * y
            one_sided_w = torch.pow(1 - pt, one_sided_gamma)
            if self.disable_torch_grad_focal_loss:
                one_sided_w = one_sided_w.detach()
            loss *= one_sided_w

        # Focal alpha
        if self.label_distribution is not None:
            label_distribution = self.label_distribution.to(device=x.device, dtype=x.dtype)
            label_total = torch.sum(label_distribution)
            if label_total <= 0:
                raise ValueError("label_distribution总数必须大于0")
            label_other = label_total - label_distribution
            # if distribution_gamma = 0 , alpha = 1
            alpha = torch.pow(label_other/label_total, self.distribution_gamma)
            loss *= alpha
        elif self.alpha is not None:
            alpha = self.alpha
            if torch.is_tensor(alpha):
                alpha = alpha.to(device=x.device, dtype=x.dtype)
            loss *= alpha

        return -loss.sum(dim=-1).mean()
