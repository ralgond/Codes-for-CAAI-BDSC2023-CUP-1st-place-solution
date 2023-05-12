import logging
from os import path
from queue import Queue
from threading import Thread

import torch
import os
import numpy as np
import random

from configure import config

def set_seed(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    # if torch.cuda.is_available():
    #     torch.cuda.manual_seed_all(seed)
    #     torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False
        
def set_logger():
    logging.basicConfig(
        format='%(asctime)s | %(levelname)s:  %(message)s',
        level=logging.INFO,
        datefmt='%Y-%m-%d %H:%M:%S',
        filename=path.join(config.save_path, 'train.log'),
        filemode='w'
    )
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    console.setFormatter(logging.Formatter('%(asctime)s | %(levelname)s:  %(message)s'))
    logging.getLogger().addHandler(console)
