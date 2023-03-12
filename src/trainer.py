from typing import List
import torch
import matplotlib.pyplot as plt
import seaborn as sns


class Trainer:
    def __init__(self, model, optimizer, loss_function, device, epochs=10):
        self.model = model
        self.optimizer = optimizer
        self.loss_function = loss_function
        self.device = device
        self.epochs = epochs

        self.train_loss: List[float] = []
        self.test_loss: List[float] = []
        self.train_accuracy: List[float] = []
        self.test_accuracy: List[float] = []

    def train(self, train_loader, epoch):
        self.model.train()

        print(f"----- Epoch {epoch} -----")

        self.train_loss.append(0)
        self.train_accuracy.append(0)

        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(self.device), target.to(self.device)

            output = self.model(data)  # get output for the input data
            loss = self.loss_function(
                output.squeeze(), target
            )  # calculate loss for the predicted output

            self.optimizer.zero_grad()  # clear gradients for this training step
            loss.backward()  # backpropagation, compute gradients
            self.optimizer.step()  # apply gradients

            self.train_loss[-1] += loss.item()
            self.train_accuracy[-1] += (output.squeeze().round() == target).sum().item()

            if batch_idx % 5000 == 0:
                loss, current = loss.item(), (batch_idx + 1) * len(data)
                print(
                    f"loss: {loss:>7f}  [{current:>5d}/{len(train_loader.dataset):>5d}]"
                )

        self.train_loss[-1] /= len(train_loader)
        self.train_accuracy[-1] *= 100.0 / len(train_loader.dataset)
        

    def test(self, test_loader):
        self.model.eval()
        test_loss = 0
        correct = 0
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)

                test_loss += self.loss_function(output.squeeze(), target).item()
                correct += (output.squeeze().round() == target).sum().item()

        self.test_loss.append(test_loss / len(test_loader))
        self.test_accuracy.append(100.0 * correct / len(test_loader.dataset))
        print(
            "Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)".format(
                self.test_loss[-1], correct, len(test_loader.dataset), self.test_accuracy[-1]
            )
        )


    def run(self, train_loader, test_loader):
        for epoch in range(1, self.epochs + 1):
            self.train(train_loader, epoch)
            self.test(test_loader)
            if len(self.test_loss)>=4 and all(self.test_loss[i]>self.train_loss[i] for i in range(-1, -5, -1)):
                print("Early stopping")
                break


    def plot_loss(self):
        """Plots train and validation (test) losses of the model in function of the epoch."""
        plt.figure()
        sns.lineplot(x=range(1, len(self.train_loss)+1), y=self.train_loss, label="Train loss")
        sns.lineplot(x=range(1, len(self.test_loss)+1), y=self.test_loss, label = "Val loss")
        plt.title("Model loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.show()

    def plot_accuracy(self):
        """Plots train and validation (test) accuracies of the model in function of the epoch."""
        plt.figure()
        sns.lineplot(x=range(1, len(self.train_accuracy)+1), y=self.train_accuracy, label = "Train accuracy")
        sns.lineplot(x=range(1, len(self.test_accuracy)+1), y=self.test_accuracy, label="Val accuracy")
        plt.title("Model accuracy")
        plt.xlabel("Epoch")
        plt.ylabel("Accuarcy")
        plt.show()
