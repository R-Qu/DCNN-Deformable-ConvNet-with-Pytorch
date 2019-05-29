import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data.sampler import SubsetRandomSampler

from SimpleCNN import *
from trainNet import *

def main() :
    seed = 42
    np.random.seed(seed)
    torch.manual_seed(seed)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    train_cifar10(device)

def train_cifar10(device):
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    train_set = torchvision.datasets.CIFAR10(root='./cifardata', train=True, download=True, transform=transform)
    test_set = torchvision.datasets.CIFAR10(root='./cifardata', train=False, download=True, transform=transform)
    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    n_training_samples = 10000
    train_sampler = SubsetRandomSampler(np.arange(n_training_samples, dtype=np.int32))
    n_val_samples = 5000
    val_sampler = SubsetRandomSampler(np.arange(n_training_samples,
                                               n_training_samples + n_val_samples, dtype=np.int32))
    n_test_samples = 5000
    batch_sizeT = 1
    test_sampler = SubsetRandomSampler(np.arange(n_test_samples, dtype=np.int32))
    train_data = torch.utils.data.DataLoader(train_set, batch_size=batch_sizeT,
                                           sampler=train_sampler, num_workers=2)
    test_data = torch.utils.data.DataLoader(test_set, batch_size=4, sampler=test_sampler, num_workers=2)
    val_data = torch.utils.data.DataLoader(train_set, batch_size=32, sampler=val_sampler, num_workers=2)
    CNN = DeformNet()
    CNN = CNN.to(device)
    trainNet(CNN, device, train_loader=train_data, val_loader = val_data,
            batch_size=batch_sizeT, n_epochs=5, learning_rate=0.001)

if __name__ == "__main__":
    main()
