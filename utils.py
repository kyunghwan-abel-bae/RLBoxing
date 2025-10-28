import random
import numpy as np
import torch

bprint_temp = [""]
def bprint(msg):
    print(msg)
    bprint_temp[0] = bprint_temp[0] + msg + "\n"

def save_bprint(str_filename):
    with open(str_filename, "a") as file:
        file.write(bprint_temp[0])
    bprint_temp[0] = ""

def clear_bprint():
    bprint_temp[0] = ""

def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

