import torch.nn.functional as F
import torch.nn as nn

class ContentLoss(nn.Module):
    """
    A module for calculating the content loss between input and target tensors.

    Args:
        target (torch.Tensor): The target tensor used for calculating the loss.

    Attributes:
        target (torch.Tensor): The target tensor used for calculating the loss.
        loss (torch.Tensor): The calculated content loss.

    """

    def __init__(self, target):
        """
        Initializes a new instance of the ContentLoss module.

        Args:
            target (torch.Tensor): The target tensor used for calculating the loss.

        """
        super(ContentLoss, self).__init__()
        self.target = target.detach()

    def forward(self, input):
        """
        Performs a forward pass through the ContentLoss module.

        Args:
            input (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The input tensor, unchanged.

        """
        self.loss = F.mse_loss(input, self.target)
        return input
