import os

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
# from nets.siamese import Siamese
from nets.siamese_repvgg import Siamese
from utils.dataloader import SiameseDataset, dataset_collate
from utils.utils_fit import fit_one_epoch


# ----------------------------------------------------#
#   计算图片总数
# ----------------------------------------------------#
def get_image_num(path, train_own_data):
    num = 0
    if train_own_data:
        train_path = os.path.join(path, 'logos')
        for character in os.listdir(train_path):
            # ----------------------------------------------------#
            #   在大众类下遍历小种类。
            # ----------------------------------------------------#
            character_path = os.path.join(train_path, character)
            num += len(os.listdir(character_path))
    else:
        train_path = os.path.join(path, 'logos')
        for alphabet in os.listdir(train_path):
            # -------------------------------------------------------------#
            #   然后遍历images_background下的每一个文件夹，代表一个大种类
            # -------------------------------------------------------------#
            alphabet_path = os.path.join(train_path, alphabet)
            for character in os.listdir(alphabet_path):
                # ----------------------------------------------------#
                #   在大众类下遍历小种类。
                # ----------------------------------------------------#
                character_path = os.path.join(alphabet_path, character)
                num += len(os.listdir(character_path))
    return num


if __name__ == "__main__":
    Cuda = True

    dataset_path = "./datasets"

    input_shape = [105, 105, 3]

    train_own_data = True

    pretrained = True

    model_path = ""

    model = Siamese(input_shape, pretrained)
    if model_path != '':
        print('Loading weights into state dict...')
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model_dict = model.state_dict()
        pretrained_dict = torch.load(model_path, map_location=device)
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if np.shape(model_dict[k]) == np.shape(v)}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)

    model_train = model.train()
    if Cuda:
        model_train = torch.nn.DataParallel(model)
        cudnn.benchmark = True
        model_train = model_train.cuda()

    loss = nn.BCELoss()

    train_ratio = 0.9
    images_num = get_image_num(dataset_path, train_own_data)
    num_train = int(images_num * train_ratio)
    num_val = images_num - num_train

    if True:
        Batch_size = 2
        Lr = 1e-4
        Init_epoch = 0
        Freeze_epoch = 50

        epoch_step = num_train // Batch_size
        epoch_step_val = num_val // Batch_size

        if epoch_step == 0 or epoch_step_val == 0:
            raise ValueError('数据集过小，无法进行训练，请扩充数据集。')

        optimizer = optim.Adam(model_train.parameters(), Lr)
        lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.96)

        train_dataset = SiameseDataset(input_shape, dataset_path, num_train, num_val, train=True,
                                       train_own_data=train_own_data)
        val_dataset = SiameseDataset(input_shape, dataset_path, num_train, num_val, train=False,
                                     train_own_data=train_own_data)
        gen = DataLoader(train_dataset, batch_size=Batch_size, num_workers=4, pin_memory=True,
                         drop_last=True, collate_fn=dataset_collate)
        gen_val = DataLoader(val_dataset, batch_size=Batch_size, num_workers=4, pin_memory=True,
                             drop_last=True, collate_fn=dataset_collate)

        for epoch in range(Init_epoch, Freeze_epoch):
            fit_one_epoch(model_train, model, loss, optimizer, epoch, epoch_step, epoch_step_val, gen, gen_val,
                          Freeze_epoch, Cuda)
            lr_scheduler.step()

    if True:
        Batch_size = 2
        Lr = 1e-5
        Freeze_epoch = 50
        Unfreeze_epoch = 100

        epoch_step = num_train // Batch_size
        epoch_step_val = num_val // Batch_size

        if epoch_step == 0 or epoch_step_val == 0:
            raise ValueError('数据集过小，无法进行训练，请扩充数据集。')

        optimizer = optim.Adam(model_train.parameters(), Lr)
        lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.96)

        train_dataset = SiameseDataset(input_shape, dataset_path, num_train, num_val, train=True,
                                       train_own_data=train_own_data)
        val_dataset = SiameseDataset(input_shape, dataset_path, num_train, num_val, train=False,
                                     train_own_data=train_own_data)
        gen = DataLoader(train_dataset, batch_size=Batch_size, num_workers=4, pin_memory=True,
                         drop_last=True, collate_fn=dataset_collate)
        gen_val = DataLoader(val_dataset, batch_size=Batch_size, num_workers=4, pin_memory=True,
                             drop_last=True, collate_fn=dataset_collate)

        for epoch in range(Freeze_epoch, Unfreeze_epoch):
            fit_one_epoch(
                model_train, model, loss, optimizer, epoch,
                epoch_step, epoch_step_val, gen, gen_val,
                Unfreeze_epoch, Cuda)
            lr_scheduler.step()
