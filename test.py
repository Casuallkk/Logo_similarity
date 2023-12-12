import numpy as np
import torch
import torch.backends.cudnn as cudnn
from PIL import Image
from nets.siamese_resnet import Siamese as siamese


class Siamese(object):
    _defaults = {
        "model_path": 'logs/ep011-loss0.520-val_loss0.601.pth',

        "input_shape": (105, 105, 3),

        "cuda": True
    }

    @classmethod
    def get_defaults(cls, n):
        if n in cls._defaults:
            return cls._defaults[n]
        else:
            return "Unrecognized attribute name '" + n + "'"

    # ---------------------------------------------------#
    #   初始化Siamese
    # ---------------------------------------------------#
    def __init__(self, **kwargs):
        self.__dict__.update(self._defaults)
        self.generate()

    # ---------------------------------------------------#
    #   载入模型
    # ---------------------------------------------------#
    def generate(self):
        # ---------------------------#
        #   载入模型与权值
        # ---------------------------#
        print('Loading weights into state dict...')
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = siamese(self.input_shape)
        tmp = torch.load(self.model_path, map_location=device)
        state_dict = tmp['model']
        for k, v in state_dict.items():
            print(k)
        params = model.state_dict()
        for k, v in params.items():
            print(k)