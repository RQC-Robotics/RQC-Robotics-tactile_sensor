import numpy as np
import yaml
import glob

import os
import dotenv

dotenv.load_dotenv()

from sacred import Experiment
from sacred.observers import MongoObserver, FileStorageObserver
import pymongo

with open('logger.yaml') as conf_file:
    log_config = yaml.safe_load(conf_file)

ex = Experiment(log_config['exp_name'])

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
for source_file in glob.glob("**/*.py", recursive=True):
    ex.add_source_file(source_file)
    
ex.add_source_file("video_pipeline/models_src.py")


# @ex.automain
# def send_csv(_run):

#     for file_name in glob.glob(log_config['log_path'] + "/*.csv"):
#         titles = np.loadtxt(file_name, delimiter=',', dtype=str, max_rows=1, ndmin=1)
#         data = np.loadtxt(file_name, delimiter=',', skiprows=1, ndmin=2)
#         for i, title in enumerate(titles):
#             for j in range(len(data)):
#                 _run.log_scalar(title, data[j, i])
