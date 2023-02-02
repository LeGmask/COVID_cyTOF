from torch import nn

class SimpleNeuralNetwork(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.linear_relu_stack = nn.Sequential(
			nn.Linear(input_size, input_size),
            nn.ReLU(),
			nn.Linear(input_size, 1),
            nn.Sigmoid())

    def forward(self, x):
        return self.linear_relu_stack(x)
    
    