import torch
import torch.nn as nn

class AdaAct(nn.Module):
    def __init__(self, activation_functions, initial_weights=None, bias=True):
        """
        Initialize the WeightedSumActivation module with learnable weights and bias.

        Parameters:
        - activation_functions: A list of PyTorch activation function objects (e.g., nn.ReLU()).
        - initial_weights: Optional initial values for weights (list of floats). If None, weights are initialized uniformly.
        - bias: If True, includes a learnable bias term.
        """
        super(AdaAct, self).__init__()

        self.activation_functions = nn.ModuleList(activation_functions)
        num_functions = len(activation_functions)
        
        # Initialize learnable weights
        if initial_weights is None:
            initial_weights = [1.0 / num_functions] * num_functions  # Uniform initialization
        self.weights = nn.Parameter(torch.tensor(initial_weights, dtype=torch.float32))

        # Initialize learnable bias
        if bias:
            self.bias = nn.Parameter(torch.tensor(0.0, dtype=torch.float32))
        else:
            self.bias = None

    def forward(self, x):
        """
        Forward pass for the weighted sum of activation functions with bias.

        Parameters:
        - x: The input tensor.

        Returns:
        - A tensor representing the weighted sum of activation functions applied to x, plus the bias.
        """
        output = torch.zeros_like(x)
        for activation, weight in zip(self.activation_functions, self.weights):
            output += weight * activation(x)
        
        if self.bias is not None:
            output += self.bias
        return output
