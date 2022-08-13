import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms.functional import rotate as torch_rotate
from torchvision.transforms import InterpolationMode
        


class ParamRNN(nn.Module):
    """Creates fully-connected net with certain layers"""

    def __init__(self, pressure_shape, signal_shape, hidden_layers: list[int]):
        super(ParamRNN, self).__init__()

        self.signal_shape = signal_shape
        self.pressure_shape = pressure_shape
        layers = [nn.Linear(
                pressure_shape[-1] * pressure_shape[-2] +
                2 * signal_shape[-1] * signal_shape[-2], hidden_layers[0]), nn.ReLU()] + \
                    [nn.Sequential(nn.Linear(*hidden_layers[i:i+2]), nn.ReLU()) for i in range(len(hidden_layers)-1)] + \
                        [nn.Linear(hidden_layers[-1], 30*30)]

        self.sequential = nn.Sequential(*layers)

        self.upsample = nn.Upsample(size=self.pressure_shape[-2:],
                                    mode='bilinear')

    def forward(self, previous_pressure, previous_signal, current_signal):
        x = torch.concat([
            torch.flatten(previous_pressure, 1),
            torch.flatten(previous_signal, 1),
            torch.flatten(current_signal, 1)
        ],
                         dim=-1)
        x = self.sequential(x)
        x = x.view(-1, 1, 30, 30)
        x = self.upsample(x)
        x = torch.squeeze(x, -3)

        return x


class ParamSingle(nn.Module):

    def __init__(self, pressure_shape, signal_shape, hidden_layers: list[int]):
        super(ParamRNN, self).__init__()

        self.signal_shape = signal_shape
        self.pressure_shape = pressure_shape
        layers = [nn.Linear(
                signal_shape[-1] * signal_shape[-2], hidden_layers[0]), nn.ReLU()] + \
                    [nn.Sequential(nn.Linear(*hidden_layers[i:i+2]), nn.ReLU()) for i in range(len(hidden_layers)-1)] + \
                        [nn.Linear(hidden_layers[-1], 30*30)]

        self.sequential = nn.Sequential(*layers)

        self.upsample = nn.Upsample(size=self.pressure_shape[-2:],
                                    mode='bilinear')

    def forward(self, previous_pressure, previous_signal, current_signal):
        x = torch.concat([torch.flatten(current_signal, 1)], dim=-1)
        x = self.sequential(x)
        x = x.view(-1, 1, 30, 30)
        x = self.upsample(x)
        x = torch.squeeze(x, -3)

        return x

class ParamStackNetSingle(nn.Module):
    def __init__(self, pressure_shape, signal_shape, hidden_layers: list[list[int]], frames_number, frames_interval):
        super(ParamStackNetSingle, self).__init__()
        single_layers, shared_layers = hidden_layers
        self.frames_interval, self.frames_number = frames_interval, frames_number 

        self.signal_shape = signal_shape
        self.pressure_shape = pressure_shape
        layers = [nn.Linear(
                signal_shape[-1] * signal_shape[-2], single_layers[0]), nn.ReLU()] + \
                    [nn.Sequential(nn.Linear(*single_layers[i:i+2]), nn.ReLU()) for i in range(len(single_layers)-1)]

        self.sequential_single = nn.Sequential(*layers)
        
        layers = [nn.Linear(
                single_layers[-1]*frames_number, shared_layers[0]), nn.ReLU()] + \
                    [nn.Sequential(nn.Linear(*shared_layers[i:i+2]), nn.ReLU()) for i in range(len(shared_layers)-1)] + \
                        [nn.Linear(shared_layers[-1], 32*32)]
        self.sequential_shared = nn.Sequential(*layers)

        self.upsample = nn.Upsample(size=self.pressure_shape[-2:],
                                    mode='bilinear')

    def forward(self, signals_stack):
        props = []
        signals_stack = torch.swapdims(signals_stack, 0, 1)
        signal = signals_stack[-1]
        for _ in signals_stack:
            props.append(self.sequential_single(torch.flatten(signal, 1)))
        x = torch.concat(props, dim=-1)
        x = self.sequential_shared(x)
        x = x.view(-1, 1, 32, 32)
        x = self.upsample(x)
        x = torch.squeeze(x, -3)

        return x

class TransferNet(nn.Module):
    def __init__(self, pressure_shape, signal_shape, frames_number, frames_interval):
        if frames_number != 1:
            raise "Can work only with signle image recognition"
        super(TransferNet, self).__init__()
        self.frames_interval, self.frames_number = frames_interval, frames_number 

        self.signal_shape = signal_shape
        self.pressure_shape = pressure_shape
        
        from pathlib import Path
        import sys

        path_root = Path(__file__).parents[1]
        sys.path.append(str(path_root))

        
        self.net = torch.load('data/base_model/TorchSensorNN5S_norm_deep.pt', map_location='cpu')
        
    def forward(self, signals):
        return self.net(signals)
    


class ParamInterpCNN(nn.Module):

    def __init__(self, output_shape, input_shape, hidden_layers, frames_number, frames_interval):
        assert frames_number == 1
        input_shape = input_shape[-2:]
        self.output_shape = output_shape[-2:]
        super(ParamInterpCNN, self).__init__()

        self.frames_interval, self.frames_number = frames_interval, frames_number 

        layers = [nn.Conv2d(input_shape[-2], hidden_layers[0], (5, 5), padding='same'), nn.ReLU()] +\
            [nn.Sequential(nn.Conv2d(*hidden_layers[i:i+2], (5, 5), padding='same'), nn.ReLU()) for i in range(len(hidden_layers)-1)] + \
                        [nn.Conv2d(hidden_layers[-1], 1, (5, 5), padding='same')]
        
        self.conv1 = nn.Sequential(*layers)
        
    def forward(self, x):
        x = torch.swapdims(x, -2, -3).repeat(1, 1, 64, 1)
        for i in range(x.shape[-3]):
            x[:, i] = torch_rotate(x[:, i], -180/x.shape[-3]*i, interpolation=InterpolationMode.BILINEAR)
        
            
        x = self.conv1(x)
        x = x.view(-1, *self.output_shape)
        return x

class SensorNN5S_norm_deep(nn.Module):

    def __init__(self, output_shape, input_shape, frames_number, frames_interval):
        assert frames_number == 1
        input_shape = (input_shape[-1], input_shape[-2]) 
        self.output_shape = output_shape[-2:]
        super(SensorNN5S_norm_deep, self).__init__()

        self.frames_interval, self.frames_number = frames_interval, frames_number 

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
        # x = torch.unsqueeze(x, 1)
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

class Unet(nn.Module):
    
    def __init__(self, output_shape, input_shape, frames_number, frames_interval):
        assert frames_number == 1
        input_shape = input_shape[-2:]
        self.output_shape = output_shape[-2:]
        super(Unet, self).__init__()

        self.frames_interval, self.frames_number = frames_interval, frames_number 

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
        
        x = torch.swapaxes(x, -2, -3).repeat(1, 1, 64, 1)
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
    
    def __init__(self, output_shape, input_shape, frames_number, frames_interval):
        assert frames_number == 1
        input_shape = input_shape[-2:]
        self.output_shape = output_shape[-2:]
        super(Unet1, self).__init__()

        self.frames_interval, self.frames_number = frames_interval, frames_number 
        
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
        
        x = torch.swapaxes(x, -2, -3).repeat(1, 1, 64, 1)
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

class Unet2(nn.Module):
      
    def __init__(self, output_shape, input_shape, frames_number, frames_interval):
        assert frames_number == 1
        input_shape = input_shape[-2:]
        self.output_shape = output_shape[-2:]
        super(Unet2, self).__init__()

        self.frames_interval, self.frames_number = frames_interval, frames_number 
        
        self.step1 = nn.Sequential(nn.Conv2d(input_shape[-2], 16, (3, 3), 1, 'same'),
                                   nn.ReLU(),
                                   nn.Conv2d(16, 16, (3, 3), 1, 'same'),
                                   nn.ReLU(),
                                   nn.ReLU(),
                                   nn.Conv2d(16, 16, (3, 3), 1, 'same'),
                                   nn.ReLU(),
                                   )
        self.finalConv = nn.Sequential(
                                   nn.Conv2d(32, 16, (3, 3), 1, 'same'),
                                   nn.ReLU(),
                                   nn.Conv2d(16, 16, (3, 3), 1, 'same'),
                                   nn.ReLU(),
                                   nn.Conv2d(16, 1, 1, padding='same'))
        def UnetBlocks(down_in_ch, down_out_ch, down_conv_n, up_in_ch, up_inner_ch, up_out_ch, up_conv_n):
            down_layers = [nn.Conv2d(down_in_ch, down_out_ch, (3, 3), 2, 1),
                                    nn.ReLU(),]
            for i in range(down_conv_n):
                down_layers.extend([nn.Conv2d(down_out_ch, down_out_ch, (3, 3), 1, 'same'),
                                    nn.ReLU()])
            down = nn.Sequential(*down_layers)
            
            
            up_layers = [nn.Conv2d(up_in_ch, up_inner_ch, (3, 3), 1, 'same'),
                                    nn.ReLU(),]
            for i in range(down_conv_n-1):
                up_layers.extend([nn.Conv2d(up_inner_ch, up_inner_ch, (3, 3), 1, 'same'),
                                    nn.ReLU()])
            up_layers.extend([nn.ConvTranspose2d(up_inner_ch, up_out_ch, (2, 2), 2, 0),
                                   nn.ReLU(),])
            up = nn.Sequential(*up_layers)
            return down, up
        
        self.down1, self.up1 = UnetBlocks(16, 16, 3, 32, 32, 16, 3)
        self.down2, self.up2 = UnetBlocks(16, 32, 3, 64, 32, 16, 3)
        self.down3, self.up3 = UnetBlocks(32, 64, 3, 128, 64, 32, 3)
        self.down4, self.up4 = UnetBlocks(64, 64, 3, 64, 64, 64, 3)
        
           
    def forward(self, x):
        
        x = torch.swapaxes(x, -2, -3).repeat(1, 1, 64, 1)
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
            
            
        
 
if __name__ == "__main__":

    from torch.utils.tensorboard import SummaryWriter
    from torchinfo import summary
    model = Unet2((64, 64), (4, 64), 1, 1)
    print(model)
    summary(model, (1, 4, 64), col_names=["input_size", "output_size", "num_params"], device='cpu')
    # writer = SummaryWriter('logdir')
    # writer.add_graph(
    #     model,
    #     (torch.rand(1, 5, 4, 64, device='cpu')))
    # writer.close()
