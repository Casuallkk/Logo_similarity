import torch
import torch.nn as nn

from nets.se_resnet import se_resnet50


def get_img_output_length(width, height):
    def get_output_length(input_length):
        # input_length += 6
        filter_sizes = [2, 2, 2, 2, 2]
        padding = [0, 0, 0, 0, 0]
        stride = 2
        for i in range(5):
            input_length = (input_length + 2 * padding[i] - filter_sizes[i]) // stride + 1
        return input_length

    return get_output_length(width) * get_output_length(height)


class Siamese(nn.Module):
    def __init__(self, input_shape, pretrained=False):
        super(Siamese, self).__init__()
        self.resnet = se_resnet50(pretrained)
        del self.resnet.avgpool
        del self.resnet.fc

        # VGG: flat_shape = 512 * get_img_output_length(input_shape[1], input_shape[0])
        flat_shape = 2048 * 4 * 4
        # resnet34: flat_shape = 521 * 4 * 4
        # resnet50&101: flat_shape = 2048 * 4 * 4
        self.fully_connect1 = torch.nn.Linear(flat_shape, 512)
        self.fully_connect2 = torch.nn.Linear(512, 1)

    def forward(self, x):
        x1, x2 = x
        # ------------------------------------------#
        #   我们将两个输入传入到主干特征提取网络
        # ------------------------------------------#
        residual = x1

        x1 = self.resnet.conv1(x1)
        x1 = self.resnet.bn1(x1)
        x1 = self.resnet.relu(x1)

        x1 = self.resnet.conv2(x1)
        x1 = self.resnet.bn2(x1)
        x1 = self.resnet.relu(x1)

        x1 = self.resnet.conv3(x1)
        x1 = self.resnet.bn3(x1)
        x1 = self.resnet.se(x1)

        if self.downsample is not None:
            residual = self.downsample(x1)

        x1 += residual
        x1 = self.resnet.relu(x1)

        residual = x2

        x2 = self.resnet.conv1(x2)
        x2 = self.resnet.bn1(x2)
        x2 = self.resnet.relu(x2)

        x2 = self.resnet.conv2(x2)
        x2 = self.resnet.bn2(x2)
        x2 = self.resnet.relu(x2)

        x2 = self.resnet.conv3(x2)
        x2 = self.resnet.bn3(x2)
        x2 = self.resnet.se(x2)

        if self.downsample is not None:
            residual = self.downsample(x2)

        x2 += residual
        x2 = self.resnet.relu(x2)
        # -------------------------#
        #   相减取绝对值
        # -------------------------#
        x1 = torch.flatten(x1, 1)
        x2 = torch.flatten(x2, 1)
        x = torch.abs(x1 - x2)
        """
        tmp = torch.sum(x)
        temp = [0.00]
        temp = torch.tensor(temp)
        temp = temp.cuda()
        if tmp == temp:
            return x
        else:
            # -------------------------#
            #   进行两次全连接
            # -------------------------#
            x = self.fully_connect1(x)
            x = self.fully_connect2(x)
            return x

        # -------------------------#
        #   进行两次全连接
        # -------------------------#
        """
        x = self.fully_connect1(x)
        x = self.fully_connect2(x)
        return x
