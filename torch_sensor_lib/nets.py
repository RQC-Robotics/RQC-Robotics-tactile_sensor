import torch
import torch.nn as nn
import torch.nn.functional as F


class TorchSensorNN(nn.Module):

    def __init__(self, input_shape, output_shape):
        input_size = input_shape[-1] * input_shape[-2]
        super(TorchSensorNN, self).__init__()
        self.sequential = nn.Sequential(
            nn.Flatten(start_dim=1), nn.Linear(input_size, 600), nn.ReLU(),
            nn.Linear(600, 600), nn.ReLU(), nn.Linear(600, 600), nn.ReLU(),
            nn.Linear(600, output_shape[-1] * output_shape[-2]), nn.ReLU(),
            nn.Unflatten(1, output_shape[-2:]))

    def forward(self, x):
        return self.sequential(x)


class TorchSensorNN5S_norm_deep(nn.Module):

    def __init__(self, input_shape, output_shape):
        input_shape = (input_shape[-1], input_shape[-2]) 
        self.output_shape = output_shape[-2:]
        super(TorchSensorNN5S_norm_deep, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 8, (3, 1), padding='same'),
            nn.ReLU()
        )
        
        self.block1 = nn.Sequential(
            nn.MaxPool2d((2, 1)),
            nn.LazyInstanceNorm2d(),
            nn.Conv2d(8, 64, (3, 1), padding='same'),
            nn.ReLU(),
            nn.MaxPool2d((2, 1), stride=(2, 1)),
            nn.LazyInstanceNorm2d()
        )
        self.pool1 = nn.MaxPool2d((4, 1), stride=(4, 1))
        self.pool2 = nn.MaxPool2d((4, 1), stride=(4, 1))

        self.block2 = nn.Sequential(
            nn.Conv2d(72, 128, (3, 1), padding='same'),
            nn.ReLU(),
            nn.MaxPool2d((2, 1), stride=(2, 1)),
            nn.LazyInstanceNorm2d(),
            nn.Conv2d(128, 128, (3, 1), padding='same'),
            nn.MaxPool2d((2, 1))
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(200, 64, (1, input_shape[-1])),
            nn.ReLU()
        )

        self.linear = nn.Sequential(
            nn.Linear(4*input_shape[-2], 15 * 13),
            nn.ReLU(),
            nn.Linear(15*13, 30*30),
            nn.ReLU(),
            nn.Linear(30*30, output_shape[-1]*output_shape[-2]),
            nn.ReLU()
        )

    def forward(self, x):
        x = torch.unsqueeze(x, 1)
        x = torch.swapdims(x, -1, -2)
        x = self.conv1(x)

        checkpoint1 = x
        x = self.block1(x)
        checkpoint1 = self.pool1(checkpoint1)
        x = torch.concat([x, checkpoint1], dim=1)
        
        checkpoint2 = x
        x = self.block2(x)
        checkpoint2 = self.pool2(checkpoint2)
        x = torch.cat([x, checkpoint2], 1)

        x = self.conv2(x)
        x = torch.flatten(x, start_dim=1)
        x = self.linear(x)
        x = x.view(-1, *self.output_shape)
        return x

# from torchsummary import summary

# model = TorchSensorNN5S_norm_deep((4, 64), (64, 64))

# print(model)
# summary(model, (4, 64), device='cpu')