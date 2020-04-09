import torch
import torch.nn as nn
import torch.nn.functional as F

class Face(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv3d(3, 32, kernel_size=(3,3,3))
        self.bn1 = nn.BatchNorm3d(32)
        self.pool1 = nn.MaxPool3d(kernel_size=(1,2,2))
        self.conv2 = nn.Conv3d(32, 64, kernel_size=(3,3,3))
        self.bn2 = nn.BatchNorm3d(64)
        self.pool2 = nn.MaxPool3d(kernel_size=(1,2,2),stride=(2,2,2))
        self.conv3 = nn.Conv3d(64, 128, kernel_size=(1,3,3))
        self.bn3 = nn.BatchNorm3d(128)
        self.pool3 = nn.MaxPool3d(kernel_size=(1,2,2),stride=(2,2,2))
        self.conv4 = nn.Conv3d(128, 256, kernel_size=(1,3,3))
        self.bn4 = nn.BatchNorm3d(256)
        self.pool4 = nn.MaxPool3d(kernel_size=(1,2,2),stride=(2,2,2))
        self.conv5 = nn.Conv3d(256, 256, kernel_size=(1,3,3))
        self.bn5 = nn.BatchNorm3d(256)
        self.pool5 = nn.AvgPool3d(kernel_size=(1,2,2))
        self.fc = nn.Linear(256*1*5*5,7)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.pool2(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x = F.relu(x)
        x = self.pool3(x)
        x = self.conv4(x)
        x = self.bn4(x)
        x = F.relu(x)
        x = self.pool4(x)
        x = self.conv5(x)
        x = self.bn5(x)
        x = self.pool5(x) 
        
        x = x.view(x.size(0), -1) #flatten
        x = self.fc(x)
                
        return x

class Context(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv3d(3, 32, kernel_size=(3,3,3))
        self.bn1 = nn.BatchNorm3d(32)
        self.pool1 = nn.MaxPool3d(kernel_size=(1,2,2))
        self.conv2 = nn.Conv3d(32, 64, kernel_size=(3,3,3))
        self.bn2 = nn.BatchNorm3d(64)
        self.pool2 = nn.MaxPool3d(kernel_size=(1,2,2),stride=(2,2,2))
        self.conv3 = nn.Conv3d(64, 128, kernel_size=(1,3,3))
        self.bn3 = nn.BatchNorm3d(128)
        self.pool3 = nn.MaxPool3d(kernel_size=(1,2,2),stride=(2,2,2))
        self.conv4 = nn.Conv3d(128, 256, kernel_size=[1,3,3])
        self.bn4 = nn.BatchNorm3d(256)
        self.pool4 = nn.MaxPool3d(kernel_size=(1,2,2),stride=(2,2,2))
        self.conv5 = nn.Conv3d(256, 256, kernel_size=(1,3,3))
        self.bn5 = nn.BatchNorm3d(256)
        self.pool5 = nn.AvgPool3d(kernel_size=(1,2,2))
        self.fc = nn.Linear(256*1*5*5,7)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.pool1(x) 
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.pool2(x) 
        x = self.conv3(x)
        x = self.bn3(x)
        x = F.relu(x)
        x = self.pool3(x)
        x = self.conv4(x)
        x = self.bn4(x)
        x = F.relu(x)
        x = self.pool4(x)
        x = self.conv5(x)
        x = self.bn5(x)
        x = self.pool5(x)

        x = x.view(x.size(0), -1) #flatten
        x = self.fc(x)
                
        return x

class Audio(nn.Module):
    def __init__(self):
        super().__init__()
                
        layer_sizes=[26,512,256,7]
        layer_sizes1=[512,256]
        self.fc= nn.ModuleList([nn.Linear(layer_sizes[i-1],layer_sizes[i]) for i in range(1,len(layer_sizes))])
        self.bn= nn.ModuleList([nn.BatchNorm1d(layer_sizes1[i]) for i in range(0,len(layer_sizes1))])

        for m in self.modules():
          if isinstance(m,nn.Linear):
            nn.init.kaiming_normal_(m.weight)
            nn.init.zeros_(m.bias)
                 
    def forward(self, x):
        x= x.view(x.size(0),-1)
        for i in range(0,len(self.fc)-1):
          x=torch.relu(self.bn[i](self.fc[i](x)))

        x= self.fc[-1](x)
        
        return x

class Model(nn.Module):
    def __init__(self, face,context,audio):
        super(Model, self).__init__()
        self.modelA = face
        self.modelB = context
        self.modelC = audio
        layer_sizes=[21,512,256,7]
        layer_sizes1=[512,256]
        self.fc= nn.ModuleList([nn.Linear(layer_sizes[i-1],layer_sizes[i]) for i in range(1,len(layer_sizes))])
        self.bn= nn.ModuleList([nn.BatchNorm1d(layer_sizes1[i]) for i in range(0,len(layer_sizes1))])
        self.softmax = nn.LogSoftmax(dim=1)
        
    def forward(self, x1, x2,audio):
        x1 = self.modelA(x1)
        x2 = self.modelB(x2)
        audio = self.modelC(audio)

        x = torch.cat((x1, x2,audio), dim=1)
        x= x.view(x.size(0),-1)
        
        for i in range(0,len(self.fc)-1):
            x=torch.relu(self.bn[i](self.fc[i](x)))

        x= self.fc[-1](x)
        x= self.softmax(x)

        return x