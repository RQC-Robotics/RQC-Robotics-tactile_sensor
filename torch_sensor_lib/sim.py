import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms.functional import rotate as torch_rotate
from torchvision.transforms import InterpolationMode
import numpy as np
from torch_sensor_lib.visual import visual_picture
# import torch_sensor_lib.vi as ds


class FiberSimulator():

    def __init__(self, config, device='cpu', signal_noise_seed=42):

        self.device = device
        self.dtype = np.float32
        self.derivative_kernel = torch.from_numpy(
            np.expand_dims([1, -2, 1], (0, 1, 3)).astype(self.dtype)).to(device)
        self.config = config
        torch.manual_seed(signal_noise_seed)
        geom = self.config['env']['sen_geometry']
        self.phys = self.config['env']['phys']
        self.alpha = self.phys['kof']
        self.n_angles = geom['n_angles']

        x = geom['distance']
        fwhm = self.phys['fiber_sensibility']['value']
        fwhm = fwhm / x
        gauss_kernel_size = self.phys['fiber_sensibility']['accuracy']
        # gauss_kernel_size = min(int(gauss_kernel_size * fwhm), image_size)

        x = np.arange(0, gauss_kernel_size, 1, self.dtype)
        y = x[:, np.newaxis]
        x0 = y0 = gauss_kernel_size // 2
        gauss = np.exp(-4 * np.log(2) * ((x - x0)**2 + (y - y0)**2) / fwhm**2)
        self.gauss_kernel = torch.from_numpy(np.expand_dims(gauss, (0, 1))).to(
            self.device)

        self.test = self.config['sim']['test_mod']

    def second_derivative(self, input):
        return F.conv2d(input, self.derivative_kernel)

    def trans_fun(self, input):
        """Reproduces experimental transmission curve"""
        return 1 - torch.sin(self.alpha * torch.minimum(
            torch.square(self.second_derivative(input)),
            torch.Tensor([2.467401]).to(self.device)))

    def sum_fiber_losses(self, input):
        return 1 - torch.prod(self.trans_fun(input), dim=-2)

    def rotate(self, input):
        return torch.concat([
            torch_rotate(input.unsqueeze(1),
                         180 / self.n_angles * i,
                         interpolation=InterpolationMode.BILINEAR)
            for i in range(self.n_angles)
        ],
                            dim=1).view(-1, 1, *input.shape[-2:])

    def fiber_real_sim(self, pressure_mat):
        """Produces outputs of sensor 

        Params:
        mas: torch.Tensor pressure maps [batch_size, H, W]

        Result:
        torch.Tensor : sensor outputs [batch_size, n_angles, W]
        """

        rot_tensor = self.rotate(pressure_mat)

        blurred_mat = F.conv2d(rot_tensor, self.gauss_kernel, padding='same')

        if self.test:
            print("Rot tensors")
            visual_picture(rot_tensor, self.n_angles)
            print("After blur")
            visual_picture(blurred_mat, self.n_angles)

        loss_tensor = self.sum_fiber_losses(blurred_mat).view(
            -1, self.n_angles, blurred_mat.shape[-1])

        std = self.phys['relative_noise']
        delt = self.phys['noise']

        signal = torch.normal(
            loss_tensor *
            torch.normal(1, std, loss_tensor.shape).to(self.device),
            std=delt)
        if self.test:
            print("Loss in fiber")
            visual_picture(1 - self.trans_fun(rot_tensor), self.n_angles)
            print("Loss sums")
            visual_picture(loss_tensor, self.n_angles, dim=1)
            print("Signal")
            visual_picture(signal, self.n_angles, dim=1)

        return signal
