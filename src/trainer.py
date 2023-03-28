from typing import List
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from torchmetrics.classification import BinaryF1Score
from tqdm import tqdm


class Trainer:
    """
    Trains a neural network using pytorch library.
    Gives access to evaluations of the training (loss, accuracy, f1 score).
    """

    def __init__(
        self,
        model,
        optimizer,
        loss_function,
        device,
        epochs=10,
        L1_regularization=False,
        L1_lambda=0.001,
        silent=False,
    ):
        """ ""
        Creates a new Trainer instance.

        :param model: The neural network model
        :param optimizer: The optimizer
        :param loss_function: The loss function
        :param device: The device's processor on which the training will be done
        :param epochs: The training's maximum number of epochs (in the case where there is no early stopping)
        :param L1_regularization: Boolean to indicate whether L1 (lasso) regularization should be applied (True) or not (False)
        :param L1_lambda: Regularization hyperparameter (determines how severe the L1 penalty is)
        :param silent: Boolean to indicate whether data about the ongoing training should be printed or not
        """
        self.model = model
        self.optimizer = optimizer
        self.loss_function = loss_function
        self.device = device
        self.epochs = epochs
        self.L1_regularization = L1_regularization
        self.L1_lambda = L1_lambda
        self.silent = silent

        self.train_loss: List[float] = []
        self.test_loss: List[float] = []
        self.train_accuracy: List[float] = []
        self.test_accuracy: List[float] = []
        self.test_f1: List[float] = []

        self.F1 = BinaryF1Score().to(self.device)

    def train(self, train_loader, epoch):
        """
        Effectuates one training epoch on the model.

        :param train_loader: The train dataloader
        :param epoch: The epoch
        """
        self.model.train()

        if not self.silent:
            print(f"----- Epoch {epoch} -----")

        self.train_loss.append(0)
        self.train_accuracy.append(0)

        for data, target in tqdm(train_loader, desc="Training", disable=self.silent):
            data, target = data.to(self.device), target.to(self.device)

            output = self.model(data)  # get output for the input data
            loss = self.loss_function(
                output.squeeze(), target
            )  # calculate loss for the predicted output

            self.train_loss[-1] += loss.item()

            if self.L1_regularization:
                 for model_param_name, model_param_value in self.model.named_parameters():
                    if model_param_name.endswith('weight'):
                        loss += self.L1_lambda * model_param_value.abs().sum()

            self.optimizer.zero_grad()  # clear gradients for this training step
            loss.backward()  # backpropagation, compute gradients
            self.optimizer.step()  # apply gradients

            self.train_accuracy[-1] += (output.squeeze().round() == target).sum().item()

        self.train_loss[-1] /= len(train_loader)
        self.train_accuracy[-1] *= 100.0 / len(train_loader.dataset)

        if not self.silent:
            print(
                "Train set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)".format(
                    self.train_loss[-1],
                    self.train_accuracy[-1] * len(train_loader.dataset) / 100,
                    len(train_loader.dataset),
                    self.train_accuracy[-1],
                )
            )

    def test(self, test_loader):
        """
        Tests the model and produces several validation data (loss, accuracy, f1 score).

        :param test_loader: The test dataloader
        """
        self.model.eval()
        test_loss = 0
        correct = 0
        F1Score = 0
        with torch.no_grad():
            for data, target in tqdm(test_loader, desc="Testing", disable=self.silent):
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)

                test_loss += self.loss_function(output.squeeze(), target).item()
                correct += (output.squeeze().round() == target).sum().item()
                F1Score += self.F1(output.squeeze(), target)

        self.test_loss.append(test_loss / len(test_loader))
        self.test_accuracy.append(100.0 * correct / len(test_loader.dataset))
        self.test_f1.append(float(F1Score / len(test_loader)))

        if not self.silent:
            print(
                "Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%), F1: {:.4f}".format(
                    self.test_loss[-1],
                    correct,
                    len(test_loader.dataset),
                    self.test_accuracy[-1],
                    self.test_f1[-1],
                )
            )

    def run(self, train_loader, test_loader):
        """
        Runs several epochs (until early stopping or until the chosen maximum number of epochs is reached) on the train and test dataloaders.

        :param train_loader: The train dataloader
        :param test_loader: The test dataloader
        """
        with tqdm(range(1, self.epochs + 1), disable=(not self.silent)) as epochs:
            for epoch in epochs:
                self.train(train_loader, epoch)
                self.test(test_loader)

                epochs.set_postfix(
                    {
                        "Loss": "{:.4f}".format(self.test_loss[-1]),
                        "Accuracy": "{:.0f}%".format(self.test_accuracy[-1]),
                        "F1": "{:.4f}".format(self.test_f1[-1]),
                    }
                )

                if len(self.test_loss) >= 6 and all(
                    self.test_loss[i - 1] <= self.test_loss[i] for i in range(-5, 0)
                ):
                    print("Early stopping")
                    break

                # Other way of stopping early
                # if len(self.test_loss)>=4 and all(self.test_loss[i]>self.train_loss[i] for i in range(-1, -5, -1)):
                #    print("Early stopping")
                #    break

    def plot_loss(self):
        """
        Plots train and validation (test) losses of the model in function of the epoch.
        """
        plt.figure()
        sns.lineplot(
            x=range(1, len(self.train_loss) + 1), y=self.train_loss, label="Train loss"
        )
        sns.lineplot(
            x=range(1, len(self.test_loss) + 1), y=self.test_loss, label="Val loss"
        )
        plt.title("Model's loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.show()

    def plot_accuracy(self):
        """
        Plots train and validation (test) accuracies of the model in function of the epoch.
        """
        plt.figure()
        sns.lineplot(
            x=range(1, len(self.train_accuracy) + 1),
            y=self.train_accuracy,
            label="Train accuracy",
        )
        sns.lineplot(
            x=range(1, len(self.test_accuracy) + 1),
            y=self.test_accuracy,
            label="Val accuracy",
        )
        plt.title("Model's accuracy")
        plt.xlabel("Epoch")
        plt.ylabel("Accuarcy")
        plt.show()

    def plot_f1(self):
        """
        Plots validation (test) F1-scores of the model in function of the epoch.
        """
        plt.figure()
        sns.lineplot(
            x=range(1, len(self.test_f1) + 1),
            y=self.test_f1,
        )
        plt.title("Model's F1 score")
        plt.xlabel("Epoch")
        plt.ylabel("Val F1 score")
        plt.show()
