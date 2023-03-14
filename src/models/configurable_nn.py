from typing import List
import matplotlib.pyplot as plt
import numpy as np
from torch import nn



class ConfigurableNeuralNetwork(nn.Module):
    def __init__(
        self, layers: List[int], dropout: float = 0.0, dropout_indexs: List[int] = []
    ):
        super().__init__()

        self.dropout = nn.Dropout(dropout)
        self.dropout_indexs = dropout_indexs

        self.layers = nn.Sequential()
        self.create_layers(layers)

    def forward(self, x):
        return self.layers(x)

    def create_layers(self, layers: List[int]):  # [61, 500, 500, 1]
        for i in range(1, len(layers) - 1):
            self.layers.append(nn.Linear(layers[i - 1], layers[i]))
            self.layers.append(nn.ReLU())
            if i in self.dropout_indexs:
                self.layers.append(self.dropout)

        self.layers.append(nn.Linear(layers[-2], layers[-1]))
        self.layers.append(nn.Sigmoid())

    def get_weights(self, layer: int):
        return self.layers[layer].weight.squeeze().cpu().detach().numpy()
    
    def plot_weights(self, layer: int, data: List[str] = []):

        print(self.get_weights(layer))
        plt.bar(data if data else range(len(self.get_weights(layer))), np.abs(self.get_weights(layer)))

