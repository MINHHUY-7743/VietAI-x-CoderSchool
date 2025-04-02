import torch
import torch.nn as nn

class MyLeNet(nn.Module):
    def __init__(self, num_class=10):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, padding=2),
            nn.Sigmoid(),
            nn.AvgPool2d(kernel_size=2, stride=2)

        )
        # self.conv1 = nn.Conv2d(in_channels=1, out_channels=6,kernel_size=5, padding=2)
        # self.act1 = nn.Sigmoid()
        # self.pool1 = nn.AvgPool2d(kernel_size=2, stride=2)

        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5),
            nn.Sigmoid(),
            nn.AvgPool2d(kernel_size=2, stride=2)

        )
        # self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5)
        # self.act2 = nn.Sigmoid()
        # self.pool2 = nn.AvgPool2d(kernel_size=2, stride=2)

        # torch.Size([1, 16, 5, 5])
        self.fc1 = nn.Linear(in_features=16*5*5, out_features=120)
        self.act3 = nn.Sigmoid()
        self.fc2 = nn.Linear(in_features=120, out_features=84)
        self.act4 = nn.Sigmoid()
        self.fc3 = nn.Linear(in_features=84, out_features=10)

    def forward(self, x):
        x = self.conv1(x)
        # x = self.act1(x)
        # x = self.pool1(x)
        x = self.conv2(x)
        # x = self.act2(x)
        # x = self.pool2(x)

        # b, c, h, w = x.shape()
        # x = x.view(b, c*h*w)
        x = x.view(x.shape[0], -1)

        x = self.fc1(x)
        x = self.act3(x)
        x = self.fc2(x)
        x = self.act4(x)
        x = self.act3(x)

        return x

model = MyLeNet(num_class=10)
sample_input = torch.randn(1, 1,28, 28)

output = model(sample_input)
print(output.shape)