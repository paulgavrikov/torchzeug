import torch 


class NormalizedModel(torch.nn.Module):
    """
    A wrapper for a model that normalizes the input before passing it to the model.
    """

    def __init__(self, model, mean, std):
        """
        Initialize the normalized model.

        Args:
            model: The model to wrap.
            mean: The mean to normalize the input with.
            std: The standard deviation to normalize the input with.
        """
        super(NormalizedModel, self).__init__()
        self.model = model
        self.mean = torch.nn.Parameter(torch.Tensor(mean).view(-1, 1, 1), requires_grad=False)
        self.std = torch.nn.Parameter(torch.Tensor(std).view(-1, 1, 1), requires_grad=False)

    def forward(self, x):
        out = (x - self.mean) / self.std 
        out = self.model(out)
        return out
