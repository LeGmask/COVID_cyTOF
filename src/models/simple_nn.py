from torch import nn


class SimpleNeuralNetwork(nn.Module):
    """
    Simple neural network containing one hidden layer and the following functions: Linear - ReLU - Linear - Sigmoid.
    """
    def __init__(self, input_size, hidden_neurons, output_size):
        """
        Creates a new SimpleNeuralNetwork instance.
         
        :param input_size: Number of neurons in the first (input) layer
        :param hidden_neurons: Number of neurons in the hidden layer
        :param output_size: Number on neurons in the last (output) layer
        """
        super().__init__()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(input_size, hidden_neurons),
            nn.ReLU(),
            nn.Linear(hidden_neurons, output_size),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.linear_relu_stack(x)
