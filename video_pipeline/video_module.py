import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
import os
from os.path import join as jn
import numpy as np
from tqdm import tqdm
from PIL import Image


class Dinamic_video_dataset(Dataset):
    '''for big datasets'''

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

        for name in self.files:
            self.file_lens.append(len(np.load(jn(signal_path, name))))

    def __len__(self):
        return len(self.chains)

    def __getitem__(self, index):
        ch = self.chains[index]
        signal = np.load(jn(self.signal_path, ch[0])).astype(np.float32)
        pressure = np.load(jn(self.pressure_path, ch[0])).astype(np.float32)
        return signal[self.chain_len * ch[1]:self.chain_len * (ch[1] + 1)], \
                pressure[self.chain_len * ch[1]:self.chain_len * (ch[1] + 1)]

    def split_to_chains(self, chain_len):
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

        for name in self.files:
            if name.endswith('npz'):
                self.signal.append(
                    np.load(jn(signal_path, name))['arr_0'].astype(np.float32))
                self.pressure.append(
                    np.load(jn(pressure_path, name))['arr_0'].astype(np.float32))
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
        return signal[self.chain_len * ch[1]:self.chain_len * (ch[1] + 1)], \
                pressure[self.chain_len * ch[1]:self.chain_len * (ch[1] + 1)]

    def split_to_chains(self, chain_len):
        self.chain_len = chain_len
        self.chains = []

        for n, file in enumerate(self.files):
            self.chains.extend([
                (n, i) for i in range(self.file_lens[n] // chain_len)
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


def predict_chain_batch(model, chain_batch, initial_pressure=None):
    subdataloader = SubDataLoader(chain_batch,
                                  initial_pressure=initial_pressure)
    prediction = []
    for _ in range(1, len(subdataloader)):
        prediction.append(model(*subdataloader.get_next_batch())[:, None])
        subdataloader.set_previous_pressure(prediction[-1])

    return torch.concat(prediction, 1)


def fit_epoch(model, video_dataset, criterion, optimizer, chain_len, batch_size,
              device):
    video_dataset.split_to_chains(chain_len)
    chains_loader = DataLoader(video_dataset, batch_size=batch_size)
    running_loss = 0.0
    processed_data = 0

    for signal, pressure in tqdm(chains_loader,
                                 ncols=100,
                                 desc='fit_epoch',
                                 unit='batch'):
        signal = signal.to(device)
        pressure = pressure.to(device)
        optimizer.zero_grad()

        prediction = \
            predict_chain_batch(model, signal, initial_pressure=pressure[:, 0])
        loss = criterion(prediction, pressure[:, 1:])
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * (signal.shape[1] - 1) * signal.shape[0]
        processed_data += (signal.shape[1] - 1) * signal.shape[0]

    train_loss = running_loss / processed_data
    return train_loss


def eval_epoch(model, video_dataset, criterion, chain_len, batch_size, device):
    video_dataset.split_to_chains(chain_len)
    chains_loader = DataLoader(video_dataset, batch_size=batch_size)
    running_loss = 0.0
    processed_data = 0

    for signal, pressure in tqdm(chains_loader,
                                 ncols=100,
                                 desc='eval_epoch',
                                 unit='batch'):
        signal = signal.to(device)
        pressure = pressure.to(device)

        with torch.no_grad():
            prediction = \
                predict_chain_batch(model, signal, initial_pressure=pressure[:, 0])
            loss = criterion(prediction, pressure[:, 1:])

        running_loss += loss.item() * (signal.shape[1] - 1) * signal.shape[0]
        processed_data += (signal.shape[1] - 1) * signal.shape[0]

    train_loss = running_loss / processed_data
    return train_loss


def predict(model, signals, device, initial_pressure=None) -> np.array:
    signal = torch.tensor(signals, device=device)
    if initial_pressure is not None:
        initial_pressure = torch.tensor(initial_pressure, device=device)
    with torch.no_grad():
        pressure = predict_chain_batch(
            model, signal[None], initial_pressure=initial_pressure[None])[0]
    return pressure.cpu().numpy()

def eval_dataset(model, video_dataset: Video_dataset, criterion, batch_size, device):
    '''Counts average loss on all videos in dataset'''
    
    chains_loader = DataLoader(list(zip(video_dataset.signal, video_dataset.pressure)), batch_size=batch_size)
    running_loss = 0.0
    processed_data = 0

    for signal, pressure in tqdm(chains_loader,
                                 ncols=100,
                                 desc='eval_dataset',
                                 unit='batch'):
        signal = signal.to(device)
        pressure = pressure.to(device)

        with torch.no_grad():
            prediction = \
                predict_chain_batch(model, signal, initial_pressure=pressure[:, 0])
            loss = criterion(prediction, pressure[:, 1:])

        running_loss += loss.item() * (signal.shape[1] - 1) * signal.shape[0]
        processed_data += (signal.shape[1] - 1) * signal.shape[0]

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
    images[0].save(outpath+".gif", save_all=True, loop=0, duration=30, append_images=images[1:])