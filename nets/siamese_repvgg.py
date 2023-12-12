import torch
import torch.nn as nn

from nets.repvgg import RepVGG_A2


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
    def __init__(self,
                 input_shape,
                 pretrained=False,
                 deploy=False):
        super(Siamese, self).__init__()
        self.repvgg = RepVGG_A2(pretrained, deploy)
        self.stage0 = self.repvgg.stage0
        self.stage1 = self.repvgg.stage1
        self.stage2 = self.repvgg.stage2
        self.stage3 = self.repvgg.stage3
        self.stage4 = self.repvgg.stage4
        del self.repvgg.gap
        del self.repvgg.linear

        # flat_shape = 512 * get_img_output_length(input_shape[1], input_shape[0])
        flat_shape = 4*4*1408
        self.fully_connect1 = torch.nn.Linear(flat_shape, 512)
        self.fully_connect2 = torch.nn.Linear(512, 1)

    def forward(self, x):
        x1, x2 = x
        # ------------------------------------------#
        #   我们将两个输入传入到主干特征提取网络
        # ------------------------------------------#
        x1 = self.stage0(x1)
        x1 = self.stage1(x1)
        x1 = self.stage2(x1)
        x1 = self.stage3(x1)
        x1 = self.stage4(x1)

        x2 = self.stage0(x2)
        x2 = self.stage1(x2)
        x2 = self.stage2(x2)
        x2 = self.stage3(x2)
        x2 = self.stage4(x2)
        # -------------------------#
        #   相减取绝对值
        # -------------------------#
        x1 = torch.flatten(x1, 1)
        x2 = torch.flatten(x2, 1)
        x = torch.abs(x1 - x2)
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
        """
        # -------------------------#
        #   进行两次全连接
        # -------------------------#
        x = self.fully_connect1(x)
        x = self.fully_connect2(x)
        return x
        """