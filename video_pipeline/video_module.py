import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
import os
from os.path import join as jn
import numpy as np
from tqdm import tqdm
from PIL import Image


class Dynamic_video_dataset(Dataset):
    '''for big datasets'''

    class SliceableDataset:

        def __init__(self, files_path, files):
            self.files_path = files_path
            self.files = files
            
        def __getitem__(self, key):
            if isinstance(key, slice):
                res = []
                for file_name in self.files[key]:
                    res.append(np.load(jn(self.files_path, file_name)))
                    if file_name.endswith('.npz'):
                        res[-1] = res[-1]['arr_0']
                return np.array(res)
            else:
                file_name = self.files[key]
                res = np.load(jn(self.files_path, file_name))
                if file_name.endswith('.npz'):
                    res = res['arr_0']
                return res

    def __init__(self, pressure_path, signal_path):
        self.pressure_path = pressure_path
        self.signal_path = signal_path
        self.files = []
        self.file_lens = []
        self.chains = []
        self.chain_len = None
        pic_path_len = len(os.path.normpath(signal_path)) + 1

        for path, folders, files in os.walk(signal_path):
            for file_name in files:
                relative_path = path[pic_path_len:]
                self.files.append(jn(relative_path, file_name))

        self.pressure = self.SliceableDataset(pressure_path, self.files)
        self.signal = self.SliceableDataset(signal_path, self.files)
        
    def __len__(self):
        return len(self.chains)

    def __getitem__(self, index):
        ch = self.chains[index]
        signal = np.load(jn(self.signal_path, ch[0])).astype(np.float32)
        pressure = np.load(jn(self.pressure_path, ch[0])).astype(np.float32)
        return signal[self.chain_len * ch[1]:self.chain_len * (ch[1] + 1)], \
                pressure[self.chain_len * ch[1]:self.chain_len * (ch[1] + 1)]

    # deprecate
    def split_to_chains(self, chain_len):
        if len(self.file_lens) == 0:
            for name in self.files:
                self.file_lens.append(len(np.load(jn(self.signal_path, name))))
        self.chain_len = chain_len
        self.chains = []

        for n, file in enumerate(self.files):
            self.chains.extend([
                (file, i) for i in range(self.file_lens[n] // chain_len)
            ])


class Video_dataset(Dataset):

    def __init__(self, pressure_path, signal_path):
        self.pressure_path = pressure_path
        self.signal_path = signal_path
        self.files = []
        self.pressure = []
        self.signal = []
        self.file_lens = []
        self.chains = []
        self.chain_len = None
        pic_path_len = len(os.path.normpath(signal_path)) + 1

        for path, folders, files in os.walk(signal_path):
            for file_name in files:
                relative_path = path[pic_path_len:]
                self.files.append(jn(relative_path, file_name))

        for name in tqdm(self.files, leave=False, desc="Dataset loading", unit='video', dynamic_ncols=True):
            if name.endswith('npz'):
                self.signal.append(
                    np.load(jn(signal_path, name))['arr_0'].astype(np.float32))
                self.pressure.append(
                    np.load(jn(pressure_path,
                               name))['arr_0'].astype(np.float32))
            else:
                self.signal.append(
                    np.load(jn(signal_path, name)).astype(np.float32))
                self.pressure.append(
                    np.load(jn(pressure_path, name)).astype(np.float32))
            self.file_lens.append(len(self.signal[-1]))

    def __len__(self):
        return len(self.chains)

    def __getitem__(self, index):
        ch = self.chains[index]
        signal = self.signal[ch[0]]
        pressure = self.pressure[ch[0]]
        return signal[(self.chain_len-1) * ch[1]:(self.chain_len-1) * ch[1] + self.chain_len], \
                pressure[(self.chain_len-1) * ch[1]:(self.chain_len-1) * ch[1] + self.chain_len]

    def split_to_chains(self, chain_len):
        self.chain_len = chain_len
        self.chains = []

        for n, file in enumerate(self.files):
            self.chains.extend([
                (n, i) for i in range((self.file_lens[n]-1) // (chain_len-1))
            ])


class SubDataLoader():

    def __init__(self, chains: torch.Tensor, initial_pressure=None):
        self.data = chains
        self.i = 0
        if initial_pressure is None:
            shape = chains.shape
            self.previous_pressure = torch.zeros(shape[0], shape[3], shape[3])
        else:
            self.previous_pressure = initial_pressure

    def __len__(self):
        return self.data.shape[1]

    def get_next_batch(self):
        self.i += 1
        if self.i > len(self):
            raise StopIteration
        return (self.previous_pressure, self.data[:, self.i - 1],
                self.data[:, self.i])

    def set_previous_pressure(self, pressure):
        self.previous_pressure = pressure


def predict_chain_batch_old(model, chain_batch, initial_pressure=None):
    subdataloader = SubDataLoader(chain_batch,
                                  initial_pressure=initial_pressure)
    prediction = []
    for _ in range(1, len(subdataloader)):
        prediction.append(model(*subdataloader.get_next_batch())[:, None])
        subdataloader.set_previous_pressure(prediction[-1])

    return torch.concat(prediction, 1)

def predict_chain_batch(model, chain_batch, heat_up=0):
    prediction = []
    if hasattr(model, "reset_memory"):
        model.reset_memory()
        
    for i in range(chain_batch.shape[1]):
        prediction.append(model(chain_batch[:, i])[:, None])
        if hasattr(model, "reset_memory"):
            model.reset_memory()
    
    return torch.concat(prediction[heat_up:], 1)


def fit_epoch(model, video_dataset, criterion, optimizer, chain_len, batch_size,
              device, heat_up=0):
    video_dataset.split_to_chains(chain_len)
    chains_loader = DataLoader(video_dataset, batch_size=batch_size)
    running_loss = 0.0
    processed_data = 0

    for signal, pressure in tqdm(chains_loader,
                                 ncols=100,
                                 desc='fit_epoch',
                                 unit='batch',
                                 leave=False,
                                 position=1,
                                 ):
        signal = signal.to(device)
        pressure = pressure.to(device)
        optimizer.zero_grad()

        prediction = \
            predict_chain_batch(model, signal, heat_up=heat_up)
        loss = criterion(prediction, pressure[:, -prediction.shape[1]:])
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * prediction.shape[1] * prediction.shape[0]
        processed_data += prediction.shape[1] * prediction.shape[0]

    train_loss = running_loss / processed_data
    return train_loss


def eval_epoch(model, video_dataset, criterion, chain_len, batch_size, device, heat_up=0):
    video_dataset.split_to_chains(chain_len)
    chains_loader = DataLoader(video_dataset, batch_size=batch_size)
    running_loss = 0.0
    processed_data = 0

    for signal, pressure in tqdm(chains_loader,
                                 ncols=100,
                                 desc='eval_epoch',
                                 unit='batch',
                                 leave=False,
                                 position=1):
        signal = signal.to(device)
        pressure = pressure.to(device)

        with torch.no_grad():
            prediction = \
                predict_chain_batch(model, signal, heat_up=heat_up)
            loss = criterion(prediction, pressure[:, -prediction.shape[1]:])

        running_loss += loss.item() * prediction.shape[1] * prediction.shape[0]
        processed_data += prediction.shape[1] * prediction.shape[0]

    train_loss = running_loss / processed_data
    return train_loss


def predict(model, signals, device) -> np.array:
    signal = torch.tensor(signals, device=device)
    with torch.no_grad():
        pressure = predict_chain_batch(
            model, signal[None])[0]
    return pressure.cpu().numpy()


def eval_dataset(model, video_dataset: Video_dataset, criterion, batch_size,
                 device):
    '''Counts average loss on all videos in dataset'''

    chains_loader = DataLoader(list(
        zip(video_dataset.signal, video_dataset.pressure)),
                               batch_size=batch_size)
    running_loss = 0.0
    processed_data = 0

    for signal, pressure in tqdm(chains_loader,
                                 ncols=100,
                                 desc='eval_dataset',
                                 unit='batch',
                                 leave=False,
                                 position=1):
        signal = signal.to(device)
        pressure = pressure.to(device)

        with torch.no_grad():
            prediction = \
                predict_chain_batch(model, signal)
            loss = criterion(prediction, pressure[:, -prediction.shape[1]:])

        running_loss += loss.item() * prediction.shape[1] * prediction.shape[0]
        processed_data += prediction.shape[1] * prediction.shape[0]

    train_loss = running_loss / processed_data
    return train_loss


# unfortunately doesn't work properly
#
# def visual_chains(list_of_chains, outpath):
#     '''
#     Params:
#     list_of_chains: List[np.array-s]
#     outpath: (str) path for avi video
#     '''
#     chains = np.concatenate(list_of_chains, -2)
#     # norming
#     chains = ((chains - chains.min()) * 255 /
#               (chains.max() - chains.min())).astype(np.uint8)
#     video = cv2.VideoWriter(outpath, 0, 10, chains.shape[-2:], 0)
#     for i in range(len(chains)):
#         video.write(cv2.cvtColor(chains[i], cv2.COLOR_RGB2GRAY))
#         cv2.imshow('a', chains[i])
#         #waits for user to press any key
#         #(this is necessary to avoid Python kernel form crashing)
#         cv2.waitKey(0)
#     #closing all open windows
#     cv2.destroyAllWindows()
#     video.release()


def visual_chains(list_of_chains, outpath):
    '''
    Params:
    list_of_chains: List[np.array-s]
    outpath: (str) path for gif video 
    '''
    chains = np.concatenate(list_of_chains, -1)
    # norming
    chains = ((chains - chains.min()) * 255 /
              (chains.max() - chains.min())).astype(np.uint8)
    images = []
    for i in range(len(chains)):
        images.append(Image.fromarray(chains[i]))
    images[0].save(outpath + ".gif",
                   save_all=True,
                   loop=0,
                   duration=30,
                   append_images=images[1:])
