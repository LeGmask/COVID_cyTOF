from typing import List

from torch import nn


class ConfigurableNeuralNetwork(nn.Module):
    def __init__(self, layers: List[int], dropout: float = 0.0):
        super().__init__()

        self.dropout = nn.Dropout(dropout)

        self.layers = nn.Sequential()
        self.create_layers(layers)

    def forward(self, x):
        return self.layers(x)
        

    def create_layers(self, layers: List[int]):
        for i in range(1, len(layers) - 1):
            self.layers.append(nn.Linear(layers[i - 1], layers[i]))
            self.layers.append(nn.ReLU())
            self.layers.append(self.dropout)

        self.layers.append(nn.Linear(layers[-2], layers[-1]))
        self.layers.append(nn.Sigmoid())
        self.layers.append(self.dropout)
