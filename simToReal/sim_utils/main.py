import numpy as np

dtype = np.float32

def generate_point_dataset(gaus_size_pix, dataset_len, force, grid_step):
    indent = (gaus_size_pix)//2+2
    pressure_shape = (dataset_len+2*indent, gaus_size_pix)
    
    point_pressure_value = force/grid_step**2
    
    dataset = np.zeros((dataset_len,)+pressure_shape, dtype=dtype)
    for i in range(indent, dataset.shape[1]-indent):
        dataset[i-indent, i, dataset.shape[2]//2] = point_pressure_value
    return dataset

def generate_flat_dataset(gaus_size_pix, gaus_sigma_pix, dataset_len, force, grid_step):
    
    indent = (gaus_size_pix)//2+2
    pressure_shape = (dataset_len+2*indent, 1)
    point_pressure_value = force/grid_step**2
    
    from torchvision.transforms.functional_tensor import _get_gaussian_kernel1d
    kernel = _get_gaussian_kernel1d(gaus_size_pix, gaus_sigma_pix)
    point_pressure_value *= kernel.max()

    
    dataset = np.zeros((dataset_len,)+pressure_shape, dtype=dtype)
    for i in range(indent, dataset.shape[1]-indent):
        dataset[i-indent, i, dataset.shape[2]//2] = point_pressure_value
    return dataset

def approx_sinuses(sinuses, period=None):
    def sinus_func(x, w, phi, A, C):
        return np.sin(x*w + phi)*A + C

    from scipy.optimize import curve_fit

    avg_signal = np.zeros(len(sinuses))
    sinus_ampl = np.zeros(len(sinuses))
    x = np.arange(len(sinuses[0]))
    for j, sinus in enumerate(sinuses):
        if period is None:
            period = len(sinus)
        p0 = [np.pi*2/period, 1, 0.1, np.mean(sinus)]
        popt, pcov = curve_fit(sinus_func, x, sinus, p0=p0)
        sinus_ampl[j] = np.abs(popt[2])
        avg_signal[j] = popt[3]
    
    return sinus_ampl, avg_signal

def simple_approx_sinuses(sinuses):
    avg_signal = np.zeros(len(sinuses))
    sinus_ampl = np.zeros(len(sinuses))
    for j, sinus in enumerate(sinuses):
        sinus_ampl[j] = (np.max(sinus) - np.min(sinus))/2
        avg_signal[j] = np.mean(sinus)
    return sinus_ampl, avg_signal

    
class SinusSimulator:
    
    def __init__(self, config: dict, simulator, sinus_periods=1):
        assert(config['env']['sen_geometry']['n_angles'] == 1)
        self.config = config.copy()
        self.simulator = simulator
        
        simb = simulator(config)
        ds_size = int(np.ceil(sinus_periods*config['env']['bimodal']['period']/simb.pixel_distance))
        self.dataset = generate_flat_dataset(simb.gaus_kernel_size, simb.gaus_sigma_pix, ds_size, 0.4, simb.pixel_distance)
        
    def generate_sinuses(self, intermod_matrix, forces):
        self.config['env']['bimodal']['intermode_matrix'] = intermod_matrix
        simb = self.simulator(self.config)
        
        sinuses = []
        for amplitude in forces:
            result_numbers = simb.fiber_real_sim(self.dataset*amplitude)
            sinus = result_numbers[:, 0, result_numbers.shape[-1]//2]
            sinuses.append(sinus.numpy())
            
        return sinuses
                
        
    
    def generate_charact_curves(self, intermod_matrix, forces, use_simple=True) -> np.array:
        
        sinuses = self.generate_sinuses(intermod_matrix, forces)
        
        if use_simple:
            return simple_approx_sinuses(sinuses)
        else:
            return approx_sinuses(sinuses)
        