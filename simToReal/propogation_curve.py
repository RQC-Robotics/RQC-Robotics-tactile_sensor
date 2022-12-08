import numpy as np
import matplotlib.pyplot as plt
from os.path import join as jn
import os
import yaml


from pathlib import Path
import sys
import os
path_root = Path(__file__).parents[1]
sys.path.append(str(path_root))
# os.chdir('..')


import os
import dotenv

dotenv.load_dotenv()

from sacred import Experiment
from sacred.observers import MongoObserver, FileStorageObserver
import pymongo

ex = Experiment('fiber_sim')

if "username" in os.environ and "host" in os.environ and "password" in os.environ:
    client = pymongo.MongoClient(
        username=os.environ['username'],
        password=os.environ['password'],
        host=os.environ['host'],
        port=27018,
        authSource=os.environ['database'],
        tls=True,
        tlsCAFile=
        "/usr/local/share/ca-certificates/Yandex/YandexInternalRootCA.crt",
    )
    ex.observers.append(
        MongoObserver(client=client, db_name=os.environ['database']))
else:
    ex.observers.append(FileStorageObserver('logdir'))
    input("WARNING! No password for db. Confirm logging locally")

ex.add_config('params.yaml')


@ex.automain
def send_csv(_run):

    # for file_name in glob.glob(log_config['log_path'] + "/*.csv"):
    #     titles = np.loadtxt(file_name, delimiter=',', dtype=str, max_rows=1, ndmin=1)
    #     data = np.loadtxt(file_name, delimiter=',', skiprows=1, ndmin=2)
    #     for i, title in enumerate(titles):
    #         for j in range(len(data)):
    #             _run.log_scalar(title, data[j, i])

    import torch

    k = 5

    with open('params.yaml') as conf_file:
        config = yaml.safe_load(conf_file)

    # config[]
    import torch_sensor_lib as tsl

    point_pres = np.zeros((64, 64), dtype=np.float32)
    point_pres[31, 31] = 1
    point_pres = np.repeat(np.repeat(point_pres, k, axis=-2), k, axis=-1)
    point_pres = torch.from_numpy(point_pres)

    config['env']['sen_geometry']['distance'] /= k
    sim = tsl.FiberSimulator(config)
    # sim.test = True
    plt.figure(figsize=(12, 9))
    plt.ylabel("пропускание")

    # x = np.linspace(0, 60, 200)
    x = np.linspace(0, 60, 100)
    # x = np.linspace(30, 60, 2)
    y = []
    for alpha in x:
        new_signal = sim.fiber_real_sim(point_pres*alpha)
        y.append(new_signal[0][0][int(32*k)].item())
        _run.log_scalar('amplitude', alpha)
        _run.log_scalar('propogation', 1-y[-1])
    y = np.array(y)
    plt.plot(x, 1-y, label='старая симуляция -- полное пропускание')

    # for title, data in zip(['amplitude', 'propogation'], [x, y]):
    #     for i in range(len(y)):
    #         _run.log_scalar(title, data[i])
    plt.legend()
    plt.show()
