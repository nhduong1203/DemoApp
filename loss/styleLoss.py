from utils import gram_matrix
import torch.nn.functional as F
import torch.nn as nn

class StyleLoss(nn.Module):
    """
    A module for calculating the style loss between input and target_feature tensors.

    Args:
        target_feature (torch.Tensor): The target_feature tensor (style image) used for calculating the loss.

    Attributes:
        target (torch.Tensor): The target Gram matrix used for calculating the loss.
        loss (torch.Tensor): The calculated style loss.

    """

    def __init__(self, target_feature):
        """
        Initializes a new instance of the StyleLoss module.

        Args:
            target_feature (torch.Tensor): The target feature tensor used for calculating the loss.

        """
        super(StyleLoss, self).__init__()
        self.target = gram_matrix(target_feature).detach()

    def forward(self, input):
        """
        Performs a forward pass through the StyleLoss module.

        Args:
            input (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The input tensor, unchanged.

        """
        G = gram_matrix(input)
        self.loss = F.mse_loss(G, self.target)
        return input
