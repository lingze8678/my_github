# DANN（Domain Adaptation Neural Network，域适应神经网络）是一种常用的迁移学习方法，在不同数据集之间进行知识迁移。本教程将介绍如何使用DANN算法实现在MNIST和MNIST-M数据集之间进行迁移学习。
#
# 首先，我们需要了解两个数据集：MNIST和MNIST-M。MNIST是一个标准的手写数字图像数据集，包含60000个训练样本和10000个测试样本。MNIST-M是从MNIST数据集中生成的带噪声的手写数字数据集，用于模拟真实场景下的图像分布差异。
#
# 接下来，我们将分为以下步骤来完成这个任务：
#
# 1、加载MNIST和MNIST-M数据集
# 2、构建DANN模型
# 3、定义损失函数
# 4、定义优化器
# 5、训练模型
# 6、评估模型
# 加载MNIST和MNIST-M数据集
# 首先，我们需要下载并加载MNIST和MNIST-M数据集。你可以使用PyTorch内置的数据集类来完成这项任务。

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm
import ssl
ssl._create_default_https_context = ssl._create_unverified_context



class DANNModel(nn.Module):
    def __init__(self):
        super(DANNModel, self).__init__()
        # Feature extractor
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=5),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(2),
            nn.ReLU(),
            nn.Conv2d(64, 50, kernel_size=5),
            nn.BatchNorm2d(50),
            nn.Dropout2d(),
            nn.MaxPool2d(2),
            nn.ReLU()
        )

        # Label classifier
        self.label_classifier = nn.Sequential(
            nn.Linear(50 * 4 * 4, 100),
            nn.BatchNorm1d(100),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(100, 10)
        )

        # Domain classifier
        self.domain_classifier = nn.Sequential(
            nn.Linear(50 * 4 * 4, 100),
            nn.BatchNorm1d(100),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(100, 1)
        )

    def forward(self, x, alpha):
        features = self.feature_extractor(x)
        features = features.view(-1, 50 * 4 * 4)
        label_outputs = self.label_classifier(features)

        # Domain classifier on source and target domains
        domain_outputs = self.domain_classifier(features)
        reversed_features = ReverseLayerF.apply(features, alpha)
        domain_outputs_reversed = self.domain_classifier(reversed_features)

        return label_outputs, domain_outputs, domain_outputs_reversed


class ReverseLayerF(torch.autograd.Function):
    """
    Reverses the gradients during backward propagation
    """

    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha
        return output, None


def train_dann(model, source_loader, target_loader, optimizer, criterion_cls, criterion_domain, alpha):
    model.train()

    running_loss = 0.0
    running_source_loss = 0.0
    running_target_loss = 0.0
    running_domain_loss = 0.0
    running_total = 0

    for i, (source_data, target_data) in enumerate(zip(source_loader, target_loader)):
        # Load source and target data
        source_inputs, source_labels = source_data
        target_inputs, _ = target_data
        batch_size = source_inputs.size(0)
        inputs = torch.cat((source_inputs, target_inputs), dim=0)
        inputs = inputs.cuda()

        # Set up domain labels for source and target data
        source_domain_labels = torch.zeros(batch_size).float().cuda()
        target_domain_labels = torch.ones(batch_size).float().cuda()
        domain_labels = torch.cat((source_domain_labels, target_domain_labels), dim=0)

        # Set up label and domain classifiers
        optimizer.zero_grad()
        label_outputs, domain_outputs, domain_outputs_reversed = model(inputs, alpha)

        # Compute losses
        label_loss = criterion_cls(label_outputs[:batch_size], source_labels.cuda())
        source_domain_loss = criterion_domain(domain_outputs[:batch_size].squeeze(), source_domain_labels)
        target_domain_loss = criterion_domain(domain_outputs[batch_size:].squeeze(), target_domain_labels)
        domain_loss = source_domain_loss + target_domain_loss
        reversed_domain_loss = criterion_domain(domain_outputs_reversed.squeeze(), domain_labels)
        # Backward and optimize
        total_loss = label_loss + domain_loss + reversed_domain_loss
        total_loss.backward()
        optimizer.step()

        # Update running loss and total samples processed
        running_loss += total_loss.item() * batch_size
        running_source_loss += label_loss.item() * batch_size
        running_target_loss += target_domain_loss.item() * batch_size
        running_domain_loss += domain_loss.item() * batch_size
        running_total += batch_size

    # Calculate average loss
    avg_loss = running_loss / running_total
    avg_source_loss = running_source_loss / running_total
    avg_target_loss = running_target_loss / running_total
    avg_domain_loss = running_domain_loss / running_total

    return avg_loss, avg_source_loss, avg_target_loss, avg_domain_loss


def test(model, loader, criterion_cls):
    model.eval()

    running_loss = 0.0
    running_corrects = 0
    running_total = 0

    with torch.no_grad():
        for inputs, labels in tqdm(loader):
            inputs = inputs.cuda()
            labels = labels.cuda()

            outputs, _, _ = model(inputs, alpha=0.0)
            loss = criterion_cls(outputs, labels)

            _, preds = torch.max(outputs, 1)
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)
            running_total += inputs.size(0)

    # Calculate accuracy and average loss
    accuracy = running_corrects.double() / running_total
    avg_loss = running_loss / running_total

    return accuracy, avg_loss


def main():
    # Set up data loaders for source and target domains
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    source_trainset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    source_trainloader = DataLoader(source_trainset, batch_size=64, shuffle=True, num_workers=2)
    source_testset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    source_testloader = DataLoader(source_testset, batch_size=64, shuffle=False, num_workers=2)
    target_trainset = datasets.USPS(root='./data', train=True, download=True, transform=transform)
    target_trainloader = DataLoader(target_trainset, batch_size=64, shuffle=True, num_workers=2)
    target_testset = datasets.USPS(root='./data', train=False, download=True, transform=transform)
    target_testloader = DataLoader(target_testset, batch_size=64, shuffle=False, num_workers=2)

    # Set up model, optimizer, and loss functions
    model = DANNModel().cuda()
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    criterion_cls = nn.CrossEntropyLoss()
    criterion_domain = nn.BCEWithLogitsLoss()

    # Train and evaluate the model
    num_epochs = 10
    for epoch in range(num_epochs):
        alpha = 2.0 / (1.0 + torch.exp(torch.tensor(-10 * epoch / num_epochs))) - 1.0  # Gradient reversal weight

        print(f'Epoch {epoch+1}/{num_epochs}, alpha={alpha:.2f}')
        train_loss, train_source_loss, train_target_loss, train_domain_loss = train_dann(model, source_trainloader,
                                                target_trainloader,optimizer, criterion_cls, criterion_domain, alpha)
        print(f'Train Loss: {train_loss:.4f}, Train Source Loss: {train_source_loss:.4f}, '
        f'Train Target Loss: {train_target_loss:.4f}, Train Domain Loss: {train_domain_loss:.4f}')

    # Evaluate the model on the source and target test sets
    source_acc, source_loss = test(model, source_testloader, criterion_cls)
    print(f'Source Test Loss: {source_loss:.4f}, Source Test Acc: {source_acc:.4f}')
    target_acc, target_loss = test(model, target_testloader, criterion_cls)
    print(f'Target Test Loss: {target_loss:.4f}, Target Test Acc: {target_acc:.4f}')


if __name__ == '__main__':
    main()
