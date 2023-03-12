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
        keep_loss = 0
        correct = 0
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(self.device), target.to(self.device)
             
            self.optimizer.zero_grad()  # clear gradients for this training step
            output = self.model(data)  # get output for the input data

            loss = self.loss_function(output.squeeze(), target)  # calculate loss for the predicted output
            loss.backward()  # backpropagation, compute gradients

            self.optimizer.step()  # apply gradients

            keep_loss += loss.item()

            if batch_idx % 100 == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(epoch, batch_idx * len(data),
                                                                               len(train_loader.dataset),
                                                                               100. * batch_idx / len(train_loader),
                                                                               loss.item()))
        self.train_loss.append(keep_loss/len(train_loader))
        

    def test(self, test_loader):
        self.model.eval()
        test_loss = 0
        correct = 0
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)

                test_loss = self.loss_function(output.squeeze(), target).item()

                # correct += (output.round() == target)

                output = output.squeeze()

                # print(output.shape, target.shape)
                # print(output.round(), target)

        self.test_loss.append(test_loss)
        self.test_accuracy.append(100. * correct / len(test_loader.dataset))
        print('Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(test_loss, correct,
                                                                                 len(test_loader.dataset),
                                                                                 self.test_accuracy[-1]))

    def run(self, train_loader, test_loader):
        for epoch in range(1, self.epochs + 1):
            self.train(train_loader, epoch)
            self.test(test_loader)

            # if self.train_loss[-1] > self.test_loss[-1]:
            #     break

    def plot_loss(self):
        """Plots train and val (test) losses of the model in function of the epoch."""
        plt.figure()
        sns.lineplot(x=range(1,self.epochs+1), y=self.train_loss, label = "Train loss")
        sns.lineplot(x=range(1,self.epochs+1), y=self.test_loss, label = "Test loss")
        plt.title("Model loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.show()

    def plot_accuracy(self):
        """Plots the train and val (test) accuracies of the model in function of the epoch."""
        plt.figure()
        # sns.lineplot(x=range(1,self.epochs+1), y=self.train_accuracy, label = "Train accuracy")
        sns.lineplot(x=range(1,self.epochs+1), y=self.test_accuracy,label = "Test accuracy")
        plt.title("Model accuracy")
        plt.xlabel("Epoch")
        plt.ylabel("Accuarcy")
        plt.show()