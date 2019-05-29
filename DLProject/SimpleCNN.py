import torch
import torch.nn.functional as F
from DeformConv2DTorch import *

class SimpleCNN(torch.nn.Module):
    
    #Our batch shape for input x is (3, 32, 32)
    
    def __init__(self):
        super(SimpleCNN, self).__init__()
        
        #Input channels = 3, output channels = 18
        self.conv1 = torch.nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.bn1 = torch.nn.BatchNorm2d(32)

        self.conv2 = torch.nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.bn2 = torch.nn.BatchNorm2d(64)
        
        self.conv3 = torch.nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.bn3 = torch.nn.BatchNorm2d(128)
        self.pool = torch.nn.AvgPool2d(kernel_size=2, stride=2, padding=0)

        self.out_size1 = 128 * 16 * 16;

        #128 * 16 * 16 input features, 128 output features (see sizing flow below)
        self.fc1 = torch.nn.Linear(self.out_size1, 128)
        
        #128 input features, 10 output features for our 10 defined classes
        self.fc2 = torch.nn.Linear(128, 10)
        
    def forward(self, x):
        
        #Computes the activation of the first convolution
        #Size changes from (3, 32, 32) to (32, 32, 32)
        x = F.relu(self.conv1(x))
        x = self.bn1(x)

        #Size changes from (32, 32, 32) to (64, 32, 32)
        x = F.relu(self.conv2(x))
        x = self.bn2(x)

        #Size changes from (64, 32, 32) to (128, 32, 32)
        x = F.relu(self.conv3(x))
        x = self.bn3(x)

        #Size changes from (128, 32, 32) to (128, 16, 16)
        x = self.pool(x)

        #Reshape data to input to the input layer of the neural net
        #Size changes from (128, 16, 16) to (1, 128 * 16 * 16)
        #Recall that the -1 infers this dimension from the other given dimension
        x = x.view(-1, self.out_size1)
        
        #Computes the activation of the first fully connected layer
        #Size changes from (1, 128 * 16 * 16) to (1, 128)
        x = F.relu(self.fc1(x))
        
        #Computes the second fully connected layer (activation applied later)
        #Size changes from (1, 128) to (1, 10)
        x = self.fc2(x)

        return(x)

class SimpleDCNN(torch.nn.Module):
    
    #Our batch shape for input x is (3, 32, 32)
    
    def __init__(self):
        super(SimpleDCNN, self).__init__()
        
        #Input channels = 3, output channels = 18
        self.conv1 = torch.nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.bn1 = torch.nn.BatchNorm2d(32)

        self.offset2 = torch.nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv2 = torch.nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.bn2 = torch.nn.BatchNorm2d(64)
        
        #self.offset3 = torch.nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        #self.bnoffset3 = torch.nn.BatchNorm2d(64)
        self.conv3 = torch.nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.bn3 = torch.nn.BatchNorm2d(128)
        #self.offsetr = ROIOffset2D(128)
        self.pool = torch.nn.AvgPool2d(kernel_size=2, stride=2, padding=0)

        self.out_size1 = 128 * 16 * 16;

        #128 * 16 * 16 input features, 128 output features (see sizing flow below)
        self.fc1 = torch.nn.Linear(self.out_size1, 128)
        
        #128 input features, 10 output features for our 10 defined classes
        self.fc2 = torch.nn.Linear(128, 10)
        
    def forward(self, x):
        
        #Computes the activation of the first convolution
        #Size changes from (3, 32, 32) to (32, 32, 32)
        x = F.relu(self.conv1(x))
        x = self.bn1(x)

        offsets = (self.offset2(x))
        offsets = offsets.view(offsets.shape[0], 32, 2, 32, 32)

        offsets = offsets.transpose(2, 4)
        offsets = offsets.view(-1, int(offsets.shape[2]), int(offsets.shape[3]), 
                                            int(offsets.shape[4]))

        x_shape1 = x.shape[1]
        x_shape0 = x.shape[0]
        x = x.view(-1, int(x.shape[2]), int(x.shape[3])).unsqueeze(1)
        x = F.grid_sample(x, offsets)
        #Size changes from (32, 32, 32) to (64, 32, 32)
        x = x.view(x_shape0, x_shape1, x.shape[2], x.shape[3])
        x = F.relu(self.conv2(x))
        x = self.bn2(x)

        #Size changes from (64, 32, 32) to (128, 32, 32)
        x = F.relu(self.conv3(x))
        x = self.bn3(x)
        #Size changes from (128, 32, 32) to (128, 16, 16)
        x = self.pool(x)

        #Reshape data to input to the input layer of the neural net
        #Size changes from (128, 16, 16) to (1, 128 * 16 * 16)
        #Recall that the -1 infers this dimension from the other given dimension
        x = x.view(-1, self.out_size1)
        
        #Computes the activation of the first fully connected layer
        #Size changes from (1, 128 * 16 * 16) to (1, 128)
        x = F.relu(self.fc1(x))
        
        #Computes the second fully connected layer (activation applied later)
        #Size changes from (1, 128) to (1, 10)
        x = self.fc2(x)

        return(x)

class DeformNet(nn.Module):
    def __init__(self):
        super(DeformNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)

        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)

        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)

        self.offsets = nn.Conv2d(128, 18, kernel_size=3, padding=1)
        self.conv4 = DeformConv2D(128, 128, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(128)

        self.classifier = nn.Linear(128, 10)

    def forward(self, x):
        # convs
        x = F.relu(self.conv1(x))
        x = self.bn1(x)
        x = F.relu(self.conv2(x))
        x = self.bn2(x)
        x = F.relu(self.conv3(x))
        x = self.bn3(x)
        # deformable convolution
        offsets = self.offsets(x)
        x = F.relu(self.conv4(x, offsets))
        x = self.bn4(x)

        x = F.avg_pool2d(x, kernel_size=32, stride=1).view(x.size(0), -1)
        x = self.classifier(x)

        return F.log_softmax(x, dim=1)