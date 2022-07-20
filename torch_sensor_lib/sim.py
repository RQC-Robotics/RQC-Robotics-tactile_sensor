import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms.functional import rotate as torch_rotate
from torchvision.transforms import InterpolationMode
import numpy as np
from torch_sensor_lib.visual import visual_picture
# import torch_sensor_lib.vi as ds
'''
Param requirements:

random_seed
env.sen_geometry
env.phys

'''


class FiberSimulator():

    def __init__(self, config, device='cpu'):

        self.device = device
        self.dtype = np.float32
        self.derivative_kernel = torch.from_numpy(
            np.expand_dims([1, -2, 1], (0, 1, 3)).astype(self.dtype)).to(device)
        self.config = config
        torch.manual_seed(config['random_seed'])
        geom = self.config['env']['sen_geometry']
        self.phys = self.config['env']['phys']
        self.alpha = self.phys['kof']
        self.n_angles = geom['n_angles']

        x = geom['distance']    # 0.5
        fwhm = self.phys['fiber_sensibility']['value']    # 0.5
        fwhm = fwhm / x    # 1
        gauss_kernel_size = self.phys['fiber_sensibility']['accuracy']
        # gauss_kernel_size = min(int(gauss_kernel_size * fwhm), image_size)

        x = np.arange(0, gauss_kernel_size, 1, self.dtype)
        y = x[:, np.newaxis]
        x0 = y0 = gauss_kernel_size // 2
        gauss = np.exp(-4 * np.log(2) * ((x - x0)**2 + (y - y0)**2) / fwhm**2)
        self.gauss_kernel = torch.from_numpy(np.expand_dims(gauss, (0, 1))).to(
            self.device)

        self.test = self.config['sim']['test_mod']

    def _second_derivative(self, input):
        return F.conv2d(input, self.derivative_kernel)

    def _trans_fun(self, input):
        """Reproduces experimental transmission curve"""
        return 1 - torch.sin(self.alpha * torch.minimum(
            torch.square(self._second_derivative(input)),
            torch.Tensor([np.pi/2]).to(self.device)))

    def _sum_fiber_losses(self, input):
        return 1 - torch.prod(self._trans_fun(input), dim=-2)

    def _rotate(self, input):
        return torch.concat([
            torch_rotate(input.unsqueeze(-3),
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
        if not isinstance(pressure_mat, torch.Tensor):
            pressure_mat = torch.tensor(pressure_mat, device=self.device)

        rot_tensor = self._rotate(pressure_mat)

        blurred_mat = F.conv2d(rot_tensor, self.gauss_kernel, padding='same')

        if self.test:
            print("Rot tensors")
            visual_picture(rot_tensor, self.n_angles, size=(7, 5))
            print("After blur")
            visual_picture(blurred_mat, self.n_angles, size=(7, 5))

        loss_tensor = self._sum_fiber_losses(blurred_mat).view(
            -1, self.n_angles, blurred_mat.shape[-1])

        std = self.phys['relative_noise']
        delt = self.phys['noise']

        signal = torch.normal(
            loss_tensor *
            torch.normal(1, std, loss_tensor.shape).to(self.device),
            std=delt)
        if self.test:
            print("Loss in fiber")
            visual_picture(1 - self._trans_fun(rot_tensor),
                           self.n_angles,
                           size=(7, 5))
            print("Loss sums")
            visual_picture(loss_tensor, self.n_angles, dim=1)
            print("Signal")
            visual_picture(signal, self.n_angles, dim=1)

        return signal

    def high_resolution_sim(self, pressure_mat, scale_factor: int):
        '''works with picture that is scale_factor times bigger then standard
        shape: (batch_size, x_len*scale_factor, y_len*scale_factor)'''
        gauss_kernel_size = scale_factor * (
            self.phys['fiber_sensibility']['accuracy'] - 1) + 1

        x = np.arange(0, gauss_kernel_size, 1, self.dtype)
        y = x[:, np.newaxis]
        x0 = y0 = gauss_kernel_size // 2
        gauss = np.exp(-2 * np.log(2) * ((x - x0)**2 + (y - y0)**2) /
                       scale_factor**2)
        gauss_kernel = torch.from_numpy(np.expand_dims(gauss,
                                                       (0, 1))).to(self.device)


        # then reproduce fiber_real_sim
        if not isinstance(pressure_mat, torch.Tensor):
            pressure_mat = torch.tensor(pressure_mat, device=self.device)
            
        pressure_mat *= scale_factor**2    # to keep second derivative the same

        rot_tensor = self._rotate(pressure_mat)

        blurred_mat = F.conv2d(rot_tensor, gauss_kernel, padding='same')

        if self.test:
            print("Rot tensors")
            visual_picture(rot_tensor, self.n_angles, size=(7, 5))
            print("After blur")
            visual_picture(blurred_mat, self.n_angles, size=(7, 5))

        loss_tensor = (1 - torch.float_power(
            torch.prod(self._trans_fun(blurred_mat), dim=-2),
            1 / scale_factor)).view(-1, self.n_angles,
                                    blurred_mat.shape[-1])[..., ::scale_factor]

        std = self.phys['relative_noise']
        delt = self.phys['noise']

        # take 1/scale_factor power to compensate increase in number of multipliers
        signal = torch.normal(
            loss_tensor *
            torch.normal(1, std, loss_tensor.shape).to(self.device),
            std=delt)
        if self.test:
            print("Loss in fiber")
            visual_picture(1 - self._trans_fun(rot_tensor),
                           self.n_angles,
                           size=(7, 5))
            print("Loss sums")
            visual_picture(loss_tensor, self.n_angles, dim=1)
            print("Signal")
            visual_picture(signal, self.n_angles, dim=1)

        return signal