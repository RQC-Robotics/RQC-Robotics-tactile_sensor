import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision.transforms.functional import rotate as torch_rotate
from torchvision.transforms import InterpolationMode
        


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
            nn.LazyBatchNorm2d(),
            nn.Conv2d(8, 64, (3, 1), padding='same'),
            nn.ReLU(),
            nn.MaxPool2d((2, 1), stride=(2, 1)),
            nn.LazyBatchNorm2d()
        )
        self.pool1 = nn.MaxPool2d((4, 1), stride=(4, 1))
        self.pool2 = nn.MaxPool2d((4, 1), stride=(4, 1))

        self.block2 = nn.Sequential(
            nn.Conv2d(64, 128, (3, 1), padding='same'),
            nn.ReLU(),
            nn.MaxPool2d((2, 1), stride=(2, 1)),
            nn.LazyBatchNorm2d(),
            nn.Conv2d(128, 128, (3, 1), padding='same'),
            nn.MaxPool2d((2, 1))
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(128, 64, (1, input_shape[-1])),
            nn.ReLU()
        )

        self.linear = nn.Sequential(
            nn.Linear(4*input_shape[-2], 15 * 13),
            nn.ReLU(),
            nn.Linear(15*13, 30*30),
            nn.ReLU(),
            nn.Linear(30*30, output_shape[-1]*output_shape[-2]),
            nn.LeakyReLU(0.01)
        )

    def forward(self, x):
        x = torch.unsqueeze(x, 1)
        x = torch.swapdims(x, -1, -2)
        x = self.conv1(x)

        checkpoint1 = x
        x = self.block1(x)
        checkpoint1 = self.pool1(checkpoint1)
        # x = torch.concat([x, checkpoint1], dim=1)
        
        checkpoint2 = x
        x = self.block2(x)
        checkpoint2 = self.pool2(checkpoint2)
        # x = torch.cat([x, checkpoint2], 1)

        x = self.conv2(x)
        x = torch.flatten(x, start_dim=1)
        x = self.linear(x)
        x = x.view(-1, *self.output_shape)
        return x


class ParamInterpCNN(nn.Module):

    def __init__(self, input_shape, output_shape, hidden_layers=[32, 16, 10]):
        input_shape = input_shape[-2:]
        self.output_shape = output_shape[-2:]
        super(ParamInterpCNN, self).__init__()

        layers = [nn.Conv2d(input_shape[-2], hidden_layers[0], (5, 5), padding='same'), nn.ReLU()] +\
            [nn.Sequential(nn.Conv2d(*hidden_layers[i:i+2], (5, 5), padding='same'), nn.ReLU()) for i in range(len(hidden_layers)-1)] + \
                        [nn.Conv2d(hidden_layers[-1], 1, (5, 5), padding='same')]
        
        self.conv1 = nn.Sequential(*layers)
        
    def forward(self, x):
        x = x.unsqueeze(-2).repeat(1, 1, 64, 1)
        for i in range(x.shape[-3]):
            x[:, i] = torch_rotate(x[:, i], -180/x.shape[-3]*i, interpolation=InterpolationMode.BILINEAR)
        
            
        x = self.conv1(x)
        x = x.view(-1, *self.output_shape)
        return x


class Unet(nn.Module):
    
    def __init__(self, input_shape, output_shape):
        
        input_shape = input_shape[-2:]
        self.output_shape = output_shape[-2:]
        super(Unet, self).__init__()

        # layers = [nn.Conv2d(input_shape[-2], hidden_layers[0], (5, 5), padding='same'), nn.ReLU()] +\
        #     [nn.Sequential(nn.Conv2d(*hidden_layers[i:i+2], (5, 5), padding='same'), nn.ReLU()) for i in range(len(hidden_layers)-1)] + \
        #                 [nn.Conv2d(hidden_layers[-1], 1, (5, 5), padding='same')]
        
        # self.conv1 = nn.Sequential(*layers)
        
        self.down1 = nn.Sequential(nn.Conv2d(input_shape[-2], 16, (3, 3), 2, 1),
                                   nn.ReLU())
        self.down2 = nn.Sequential(nn.Conv2d(16, 32, (3, 3), 2, 1),
                                   nn.ReLU())
        self.down3 = nn.Sequential(nn.Conv2d(32, 64, (3, 3), 2, 1),
                                   nn.ReLU())
        self.down4 = nn.Sequential(nn.Conv2d(64, 128, (3, 3), 2, 1),
                                   nn.ReLU())
        self.up4 = nn.Sequential(nn.ConvTranspose2d(128, 64, (2, 2), 2, 0),
                                   nn.ReLU())
        self.up3 = nn.Sequential(nn.ConvTranspose2d(128, 32, (2, 2), 2, 0),
                                   nn.ReLU())
        self.up2 = nn.Sequential(nn.ConvTranspose2d(64, 16, (2, 2), 2, 0),
                                   nn.ReLU())
        self.up1 = nn.Sequential(nn.ConvTranspose2d(32, 8, (2, 2), 2, 0),
                                   nn.ReLU())
        
        self.finalConv = nn.Conv2d(8, 1, (3, 3), padding='same')
        
        
    def forward(self, x):
        
        x = x.unsqueeze(-2).repeat(1, 1, 64, 1)
        for i in range(x.shape[-3]):
            x[:, i] = torch_rotate(x[:, i], -180/x.shape[-3]*i, interpolation=InterpolationMode.BILINEAR)
            
        res1 = self.down1(x)
        
        res2 = self.down2(res1)
        
        res3 = self.down3(res2)
        
        res4 = self.down4(res3)
        
        x = self.up4(res4)
        
        x = torch.concat([x, res3], -3)
        
        x = self.up3(x)
        
        x = torch.concat([x, res2], -3)
        
        x = self.up2(x)
        
        x = torch.concat([x, res1], -3)
        
        x = self.up1(x)
        
        x = self.finalConv(x)
        x = x.squeeze(-3)
        
        # x = x.view(-1, *self.output_shape)
        return x



class Unet1(nn.Module):
    
    def __init__(self, input_shape, output_shape):
        
        input_shape = input_shape[-2:]
        self.output_shape = output_shape[-2:]
        super(Unet1, self).__init__()

        self.step1 = nn.Sequential(nn.Conv2d(input_shape[-2], 16, (3, 3), 1, 'same'),
                                   nn.ReLU(),
                                   nn.Conv2d(16, 16, (3, 3), 1, 'same'),
                                   nn.ReLU(),
                                   nn.ReLU(),
                                   nn.Conv2d(16, 16, (3, 3), 1, 'same'),
                                   nn.ReLU(),
                                   )
        self.down1 = nn.Sequential(nn.Conv2d(16, 32, (3, 3), 2, 1),
                                   nn.ReLU(),
                                   nn.Conv2d(32, 32, (3, 3), 1, 'same'),
                                   nn.ReLU(),
                                   nn.Conv2d(32, 32, (3, 3), 1, 'same'),
                                   nn.ReLU(),
                                   )
        self.down2 = nn.Sequential(
                                   nn.Conv2d(32, 32, (3, 3), 2, 1),
                                   nn.ReLU(),
                                   nn.Conv2d(32, 32, (3, 3), 1, 'same'),
                                   nn.ReLU(),
                                   nn.Conv2d(32, 32, (3, 3), 1, 'same'),
                                   nn.ReLU(),
                                   )
        self.down3 = nn.Sequential(nn.Conv2d(32, 64, (3, 3), 2, 1),
                                   nn.ReLU(),
                                   nn.Conv2d(64, 64, (3, 3), 1, 'same'),
                                   nn.ReLU(),
                                   nn.Conv2d(64, 64, (3, 3), 1, 'same'),
                                    nn.ReLU(),)
        self.down4 = nn.Sequential(
                                    nn.Conv2d(64, 128, (3, 3), 2, 1),
                                    nn.ReLU(),
                                    nn.Conv2d(128, 128, (3, 3), 1, 'same'),
                                    nn.ReLU(),
                                    nn.Conv2d(128, 128, (3, 3), 1, 'same'),
                                    nn.ReLU(),
                                    nn.Conv2d(128, 128, (3, 3), 1, 'same'),
                                    nn.ReLU(),)
        self.up4 = nn.Sequential(
                                   nn.Conv2d(128, 64, (3, 3), 1, 'same'),
                                   nn.ReLU(),
                                   nn.Conv2d(64, 64, (3, 3), 1, 'same'),
                                   nn.ReLU(),
                                   nn.ConvTranspose2d(64, 32, (2, 2), 2, 0),
                                   nn.ReLU(),)
        self.up3 = nn.Sequential(
                                   nn.Conv2d(32+64, 32, (3, 3), 1, 'same'),
                                   nn.ReLU(),
                                   nn.Conv2d(32, 32, (3, 3), 1, 'same'),
                                   nn.ReLU(),
                                   nn.ConvTranspose2d(32, 16, (2, 2), 2, 0),
                                   nn.ReLU(),)
        self.up2 = nn.Sequential(
                                   nn.Conv2d(48, 16, (3, 3), 1, 'same'),
                                   nn.ReLU(),
                                   nn.Conv2d(16, 16, (3, 3), 1, 'same'),
                                   nn.ReLU(),
                                   nn.ConvTranspose2d(16, 16, (2, 2), 2, 0),
                                   nn.ReLU(),)
        self.up1 = nn.Sequential(
                                   nn.Conv2d(48, 16, (3, 3), 1, 'same'),
                                   nn.ReLU(),
                                   nn.Conv2d(16, 16, (3, 3), 1, 'same'),
                                   nn.ReLU(),nn.ConvTranspose2d(16, 16, (2, 2), 2, 0),
                                   nn.ReLU(),)
        self.finalConv = nn.Sequential(
                                   nn.Conv2d(32, 16, (3, 3), 1, 'same'),
                                   nn.ReLU(),
                                   nn.Conv2d(16, 16, (3, 3), 1, 'same'),
                                   nn.ReLU(),
                                   nn.Conv2d(16, 1, 1, padding='same'))
        
    def forward(self, x):
        
        x = x.unsqueeze(-2).repeat(1, 1, 64, 1)
        for i in range(x.shape[-3]):
            x[:, i] = torch_rotate(x[:, i], -180/x.shape[-3]*i, interpolation=InterpolationMode.BILINEAR)
            
        res0 = self.step1(x)
        res1 = self.down1(res0)
        
        res2 = self.down2(res1)
        
        res3 = self.down3(res2)
        
        x = self.down4(res3)
        x = self.up4(x)
        
        x = torch.concat([x, res3], -3)
        x = self.up3(x)
        
        x = torch.concat([x, res2], -3)
        
        x = self.up2(x)
        
        x = torch.concat([x, res1], -3)
        
        x = self.up1(x)
        
        x = torch.concat([x, res0], -3)
        
        x = self.finalConv(x)
        x = x.squeeze(-3)
        
        # x = x.view(-1, *self.output_shape)
        return x


class DeepUpConv(nn.Module):

    def __init__(self, input_shape, output_shape):
        input_shape = (input_shape[-1], input_shape[-2]) 
        self.output_shape = output_shape[-2:]
        super(DeepUpConv, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 8, (3, 1), padding='same'),
            nn.ReLU()
        )
        
        self.block1 = nn.Sequential(
            nn.MaxPool2d((2, 1)),
            nn.LazyBatchNorm2d(),
            nn.Conv2d(8, 64, (3, 1), padding='same'),
            nn.ReLU(),
            nn.MaxPool2d((2, 1), stride=(2, 1)),
            nn.LazyBatchNorm2d()
        )
        self.pool1 = nn.MaxPool2d((4, 1), stride=(4, 1))
        self.pool2 = nn.MaxPool2d((4, 1), stride=(4, 1))

        self.block2 = nn.Sequential(
            nn.Conv2d(64, 128, (3, 1), padding='same'),
            nn.ReLU(),
            nn.MaxPool2d((2, 1), stride=(2, 1)),
            nn.LazyBatchNorm2d(),
            nn.Conv2d(128, 128, (3, 1), padding='same'),
            nn.MaxPool2d((2, 1))
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(128, 64, (1, input_shape[-1])),
            nn.ReLU()
        )

        self.linear = nn.Sequential(
            nn.Linear(4*input_shape[-2], 15 * 13),
            nn.ReLU(),
            nn.Linear(15*13, 30*30),
            nn.ReLU(),
        #     nn.Linear(30*30, output_shape[-1]*output_shape[-2]),
        #     nn.LeakyReLU(0.01)
        )
        
        self.UpConv = nn.Sequential(
            nn.ConvTranspose2d(1, 1, (5, 5), padding=1), nn.ReLU(), 
            nn.ConvTranspose2d(1, 1, (4, 4), stride=2, padding=1)
            
            # , nn.ReLU(),
            # nn.ConvTranspose2d(64, 32, (3, 3), stride=(2, 2)), nn.ReLU(), 
            # nn.Conv2d(32, 32, (3, 3), padding='same'), nn.ReLU(),
            # nn.ConvTranspose2d(32, 16, (3, 3), stride=(2, 2)), nn.ReLU(),
            # nn.Conv2d(16, 16, (3, 3), padding='same'), nn.ReLU(),
            # nn.ConvTranspose2d(16, 1, (4, 4), stride=(2, 2), padding=4), nn.ReLU(),
            # nn.Conv2d(1, 1, (5, 5), padding='same')
        )

    def forward(self, x):
        x = torch.unsqueeze(x, 1)
        x = torch.swapdims(x, -1, -2)
        x = self.conv1(x)

        checkpoint1 = x
        x = self.block1(x)
        checkpoint1 = self.pool1(checkpoint1)
        # x = torch.concat([x, checkpoint1], dim=1)
        
        checkpoint2 = x
        x = self.block2(x)
        checkpoint2 = self.pool2(checkpoint2)
        # x = torch.cat([x, checkpoint2], 1)

        x = self.conv2(x)
        x = torch.flatten(x, start_dim=1)
        x = self.linear(x)
        x = x.view(-1, 1, 30, 30)
        x = self.UpConv(x)
        x = torch.squeeze(x, -3)
        return x
    



class UpSampleNet(nn.Module):

    def __init__(self, input_shape, output_shape):
        input_shape = (input_shape[-1], input_shape[-2])
        self.output_shape = output_shape[-2:]
        super(UpSampleNet, self).__init__()

        self.conv1 = nn.Sequential(nn.Conv2d(1, 8, (3, 1), padding='same'),
                                   nn.ReLU())

        self.block1 = nn.Sequential(nn.AvgPool2d((2, 1)),
                                    nn.LazyInstanceNorm2d(),
                                    nn.Conv2d(8, 64, (3, 1), padding='same'),
                                    nn.ReLU(),
                                    nn.AvgPool2d((2, 1), stride=(2, 1)),
                                    nn.LazyInstanceNorm2d())
        self.pool1 = nn.AvgPool2d((4, 1), stride=(4, 1))
        self.pool2 = nn.AvgPool2d((4, 1), stride=(4, 1))

        self.block2 = nn.Sequential(nn.Conv2d(64, 128, (3, 1), padding='same'),
                                    nn.ReLU(),
                                    nn.AvgPool2d((2, 1), stride=(2, 1)),
                                    nn.LazyInstanceNorm2d(),
                                    nn.Conv2d(128, 128, (3, 1), padding='same'),
                                    nn.AvgPool2d((2, 1)))

        self.conv2 = nn.Sequential(nn.Conv2d(128, 64, (1, input_shape[-1])),
                                   nn.ReLU())

        self.linear = nn.Sequential(nn.Linear(4 * input_shape[-2], 20 * 30),
                                    nn.ReLU(), nn.Linear(20 * 30, 30 * 30),
                                    nn.ReLU(), nn.Linear(30 * 30, 20 * 20),
                                    nn.LeakyReLU(0.01))
        self.upsample = nn.Upsample(size=self.output_shape, mode='bilinear')

    def forward(self, x):
        x = torch.unsqueeze(x, 1)
        x = torch.swapdims(x, -1, -2)
        x = self.conv1(x)

        checkpoint1 = x
        x = self.block1(x)
        checkpoint1 = self.pool1(checkpoint1)
        # x = torch.concat([x, checkpoint1], dim=1)

        checkpoint2 = x
        x = self.block2(x)
        checkpoint2 = self.pool2(checkpoint2)
        # x = torch.cat([x, checkpoint2], 1)

        x = self.conv2(x)
        x = torch.flatten(x, start_dim=1)
        x = self.linear(x)
        x = x.view(-1, 1, 20, 20)
        x = self.upsample(x)
        x = torch.squeeze(x, -3)
        return x


if __name__ == "__main__":
    from torchinfo import summary

    model = DeepUpConv((4, 64), (64, 64))

    summary(
        model, (1, 4, 64),
        device='cpu',
        col_names=["input_size", "output_size", "num_params", "kernel_size"])
