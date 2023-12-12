## 基于孪生神经网络的logo相似度计算

大二时基于大创改的一个东西。

### 项目介绍

使用改进的Siamese Network神经网络，以RepVGG为backbone，对提取出的近似Logo进行整体相似度的计算，较快、较全面地计算商标的相似度。当相似度超过某一阈值时，系统判定其为侵权。

网络架构如下：

<img title="" src="file://pictures/architecture.png" alt="" width="344" data-align="center">

### 最终效果

![](E:\Pythonfiles\srtp\Siamese_pytorch_master\pictures\results.png)

### 模型性能

综合来看，使用RepVGG作为主干网络可将模型平均检测正确率和平均检测效率分别提高8.1%和18.4%

<img title="" src="file://pictures/accuracy.png" alt="" width="441">

<img title="" src="file://pictures/time.png" alt="" width="425">

### References：

https://github.com/bubbliiiing/Siamese-pytorchhttps://github.com/bubbliiiing/Siamese-pytorch

[GitHub - tensorfreitas/Siamese-Networks-for-One-Shot-Learning: Implementation of Siamese Neural Networks for One-shot Image Recognition](https://github.com/tensorfreitas/Siamese-Networks-for-One-Shot-Learning)


