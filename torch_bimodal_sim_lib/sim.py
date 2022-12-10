import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms.functional import rotate as torch_rotate
from torchvision.transforms.functional import gaussian_blur
from torchvision.transforms import InterpolationMode
import numpy as np
from torch_sensor_lib.visual import visual_picture
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

        gaus_sigma_mm = config['env']['phys']['sigma']
        self.pixel_distance = config['env']['sen_geometry']['distance']
        self.gaus_sigma_pix = gaus_sigma_mm/self.pixel_distance
        self.gaus_kernel_size = 1 + 2*int(3*self.gaus_sigma_pix)   # approximately good formua to get odd integer
        self.elasticity = config['env']['phys']['elasticity']


        self.test = self.config['sim']['test_mod']

        self.dk = 2*np.pi/config['env']['bimodal']['period']
        C = config['env']['bimodal']['intermode_matrix']   # intermode_matrix
        self.main_loss_coeff = config['env']['bimodal']['main_loss_coeff']
        self.loss_funcs = [
            [lambda x, i=i, j=j: self.main_loss_coeff*self.pixel_distance*C[i][j]*x**2 for j in range(2)]
            for i in range(2)
        ]
        self.vector1 = np.array(config['env']['bimodal']['borning_modes']).reshape(2)
        self.vector2 = np.array(config['env']['bimodal']['impact_modes']).reshape(2)
 

    def _second_derivative(self, input):
        return F.conv2d(input, self.derivative_kernel)/self.pixel_distance**2

        
    def _sum_fiber_losses(self, curvatures):
        
        array_of_matrixes = np.zeros((curvatures.shape[-2], 2, 2), dtype=torch.Tensor)
        # (Y-2, 2, 2), contains big tensors (n_angles*n_pictures, 1, 64), that will be element-wise multipiled
        I = torch.view_as_complex(torch.Tensor([0, 1.]))
        for i in range(curvatures.shape[-2]):
            #TODO проверить, что не слишком большие штуки интегрируем
            array_of_matrixes[i, 0, 0] = 1 - self.loss_funcs[0][0](curvatures[..., i, :])
            array_of_matrixes[i, 0, 1] = self.loss_funcs[0][1](curvatures[..., i, :])
            array_of_matrixes[i, 1, 0] = self.loss_funcs[1][0](curvatures[..., i, :])
            array_of_matrixes[i, 1, 1] = (1 - self.loss_funcs[1][1](curvatures[..., i, :]))*torch.exp(I*self.pixel_distance*self.dk)
        
        result_matrixes = np.linalg.multi_dot(array_of_matrixes)
        result_numbers = 1-(self.vector2@result_matrixes@self.vector1).abs()

        return result_numbers
        
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

        rot_tensor = self._rotate(pressure_mat*self.elasticity)
        # max_shapes = [2*t for t in rot_tensor.shape[-2:][::-1]]
        max_shapes = rot_tensor.shape[-2:][::-1]
        kernel_shape = np.minimum(max_shapes, self.gaus_kernel_size)
        kernel_shape = [t if t%2==1 else t-1 for t in kernel_shape]
        blurred_mat = gaussian_blur(rot_tensor, kernel_shape, self.gaus_sigma_pix)

        curvature = self._second_derivative(blurred_mat)
        if self.test:
            print("Rot tensors")
            visual_picture(rot_tensor, self.n_angles, size=(7, 5))
            print("After blur")
            visual_picture(blurred_mat, self.n_angles, size=(7, 5))

        loss_tensor = self._sum_fiber_losses(curvature).view(
            -1, self.n_angles, blurred_mat.shape[-1])

        std = self.phys['relative_noise']
        delt = self.phys['noise']

        signal = torch.normal(
            loss_tensor *
            torch.normal(1, std, loss_tensor.shape).to(self.device),
            std=delt)
        if self.test:
            print("Loss sums")
            visual_picture(loss_tensor, self.n_angles, dim=1)
            # print("Signal")
            # visual_picture(signal, self.n_angles, dim=1)

        return signal
