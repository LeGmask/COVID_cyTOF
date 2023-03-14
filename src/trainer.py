from typing import List
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from torchmetrics.classification import BinaryF1Score
from tqdm import tqdm


class Trainer:
    def __init__(self, model, optimizer, loss_function, device, epochs=10, L1_regularization = False, L1_lambda = 0.001):
        self.model = model
        self.optimizer = optimizer
        self.loss_function = loss_function
        self.device = device
        self.epochs = epochs

        self.train_loss: List[float] = []
        self.test_loss: List[float] = []
        self.train_accuracy: List[float] = []
        self.test_accuracy: List[float] = []
        self.train_f1: List[float] = []
        self.test_f1: List[float] = []

        self.L1_regularization = L1_regularization
        self.L1_lambda = L1_lambda

        self.F1 = BinaryF1Score().to(self.device)

    def train(self, train_loader, epoch):
        self.model.train()

        print(f"----- Epoch {epoch} -----")

        self.train_loss.append(0)
        self.train_accuracy.append(0)

        for data, target in tqdm(
            train_loader,
            desc="Training",
        ):
            data, target = data.to(self.device), target.to(self.device)

            output = self.model(data)  # get output for the input data
            loss = self.loss_function(
                output.squeeze(), target
            )  # calculate loss for the predicted output

            if self.L1_regularization:
                l1_norm = sum(torch.linalg.norm(p, 1) for p in self.model.parameters())
                loss += self.L1_lambda * l1_norm

            self.optimizer.zero_grad()  # clear gradients for this training step
            loss.backward()  # backpropagation, compute gradients
            self.optimizer.step()  # apply gradients

            self.train_loss[-1] += loss.item()
            self.train_accuracy[-1] += (output.squeeze().round() == target).sum().item()

        self.train_loss[-1] /= len(train_loader)
        self.train_accuracy[-1] *= 100.0 / len(train_loader.dataset)

        print(
            "Train set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)".format(
                self.train_loss[-1],
                self.train_accuracy[-1] * len(train_loader.dataset) / 100,
                len(train_loader.dataset),
                self.train_accuracy[-1],
            )
        )

    def test(self, test_loader):
        self.model.eval()
        test_loss = 0
        correct = 0
        F1Score = 0
        with torch.no_grad():
            for data, target in tqdm(test_loader, desc="Testing"):
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)

                test_loss += self.loss_function(output.squeeze(), target).item()
                correct += (output.squeeze().round() == target).sum().item()
                F1Score += self.F1(output.squeeze(), target)

        self.test_loss.append(test_loss / len(test_loader))
        self.test_accuracy.append(100.0 * correct / len(test_loader.dataset))
        self.test_f1.append(float(F1Score / len(test_loader)))

        print(
            "Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%), F1 {:.4f}".format(
                self.test_loss[-1],
                correct,
                len(test_loader.dataset),
                self.test_accuracy[-1],
                self.test_f1[-1],
            )
        )

    def run(self, train_loader, test_loader):
        for epoch in range(1, self.epochs + 1):
            self.train(train_loader, epoch)
            self.test(test_loader)
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
        """Plots train and validation (test) losses of the model in function of the epoch."""
        plt.figure()
        sns.lineplot(
            x=range(1, len(self.train_loss) + 1), y=self.train_loss, label="Train loss"
        )
        sns.lineplot(
            x=range(1, len(self.test_loss) + 1), y=self.test_loss, label="Val loss"
        )
        plt.title("Model loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.show()

    def plot_accuracy(self):
        """Plots train and validation (test) accuracies of the model in function of the epoch."""
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
        plt.title("Model accuracy")
        plt.xlabel("Epoch")
        plt.ylabel("Accuarcy")
        plt.show()

    def plot_f1(self):
        """Plots validation (test) F1-scores of the model in function of the epoch."""
        plt.figure()
        sns.lineplot(
            x=range(1, len(self.test_f1) + 1),
            y=self.test_f1,
            label="Val F1",
        )
        plt.title("Model F1")
        plt.xlabel("Epoch")
        plt.ylabel("F1")
        plt.show()
