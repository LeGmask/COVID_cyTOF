from typing import List

import torch


class Trainer:
    def __init__(self, model, optimizer, loss_function, device, epochs=10):
        self.model = model
        self.optimizer = optimizer
        self.loss_function = loss_function
        self.device = device
        self.epochs = epochs

        self.train_loss: List[float] = []
        self.test_loss: List[float] = []

    def train(self, train_loader, epoch):
        self.model.train()
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(self.device), target.to(self.device)

            self.optimizer.zero_grad()  # clear gradients for this training step
            output = self.model(data)  # get output for the input data

            loss = self.loss_function(output, target)  # calculate loss for the predicted output
            loss.backward()  # backpropagation, compute gradients

            self.optimizer.step()  # apply gradients
            self.test_loss.append(loss.item())

            if batch_idx % 100 == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(epoch, batch_idx * len(data),
                                                                               len(train_loader.dataset),
                                                                               100. * batch_idx / len(train_loader),
                                                                               loss.item()))

    def test(self, test_loader):
        self.model.eval()
        test_loss = 0
        correct = 0
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                test_loss += self.loss_function(output, target).item()
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()

        test_loss /= len(test_loader.dataset)
        self.test_loss.append(test_loss)
        print('Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(test_loss, correct,
                                                                                 len(test_loader.dataset),
                                                                                 100. * correct / len(
                                                                                     test_loader.dataset)))

    def run(self, train_loader, test_loader):
        for epoch in range(1, self.epochs + 1):
            self.train(train_loader, epoch)
            self.test(test_loader)
