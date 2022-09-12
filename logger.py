import os
from tensorboardX import SummaryWriter

class Logger(object):
    def __init__(self, log_path : str):
        self.summary = SummaryWriter(log_path)

    
    def add(self, epoch, scalar_dict, prefix = ''):
        for tag, scalar_val in scalar_dict.items():
            self.summary.add_scalar(prefix + tag, scalar_val, epoch)


    def close(self):
        self.summary.close()