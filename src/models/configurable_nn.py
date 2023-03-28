from typing import List
import matplotlib.pyplot as plt
import numpy as np
from torch import nn
import seaborn as sns


class ConfigurableNeuralNetwork(nn.Module):
    """
    Configurable neural network (choosable number of layers and number of neurons in each layer),
    alternating Linear and ReLU functions until the last function, a Sigmoid.
    Can have neuron dropout on chosen layers.
    """

    def __init__(
        self, layers: List[int], dropout: float = 0.0, dropout_indexs: List[int] = []
    ):
        """
        Creates a new ConfigurableNeuralNetwork instance.

        :param layers: List of the model's layers' number of neurons
        :param dropout: Percentage of dropout
        :param dropout_indexs: Layers' indexes on which there will be neuron dropout
        """
        super().__init__()

        self.dropout = nn.Dropout(dropout)
        self.dropout_indexs = dropout_indexs

        self.layers = nn.Sequential()
        self.create_layers(layers)

    def forward(self, x):
        return self.layers(x)

    def create_layers(self, layers: List[int]):
        """
        Fills the layers attribute of the model.

        :param layers: List of the model's layers' number of neurons
        """
        for i in range(1, len(layers) - 1):
            self.layers.append(nn.Linear(layers[i - 1], layers[i]))
            self.layers.append(nn.ReLU())
            if i in self.dropout_indexs:
                self.layers.append(self.dropout)

        self.layers.append(nn.Linear(layers[-2], layers[-1]))
        self.layers.append(nn.Sigmoid())

    def get_weights(self, layer: int):
        """
        Returns a numpy array containing the weights of the neurons belonging to the chosen layer.

        :param layer: Layer number of the model (starting at 0)
        """
        return self.layers[layer * 2].weight.squeeze().cpu().detach().numpy()

    def plot_weights(self, layer: int = 0, features: List[str] = None):
        """
        Plots a bar graph of the chosen layer's weights in function of the features they are associated to (or neuron number if there are no features).

        :param layer: Layer number of the model (starting at 0)
        :param variables: Features associated to the layer's neurons
        """
        plt.figure(figsize=(15, 5))

        mean_weights = np.sum(np.abs(self.get_weights(layer)), axis=0) / self.get_weights(layer).shape[0]

        idx = np.argsort(mean_weights)[::-1]
        mean_weights = mean_weights[idx].tolist()

        if features:
            features = list(np.array(features)[idx])
            plt.xticks(rotation=70)
            plt.tight_layout()


        sns.barplot(
            x=features or list(range(self.get_weights(layer * 2).shape[1])),
            y=mean_weights,
        )

        plt.title("Weights per feature")
        plt.xlabel("Features")
        plt.ylabel("Absolute mean weight")
        
        plt.show()
