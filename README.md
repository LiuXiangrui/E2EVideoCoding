# E2EVideoCoding

# DVC
- 参考![PytorchCompression](https://github.com/ZhihaoHu/PyTorchVideoCompression/tree/master/DVC)实现
- 发现PytorchCompression的实现与DVC原文差别
  - MV的编码器使用ReLU而非GDN
  - MC网络没有将光流作为输入
  - 残差编码器通道数设置与DVC不同(DVC采用Balle Hyperprior的通道数设置)

# DCVC
- 网络结构与官方代码![DCVC](https://github.com/microsoft/DCVC)一致

# FVC
- 根据论文进行实现
- 论文未描述清楚地方：
  - 图1中参考帧与当前帧的特征提取是否权值共享，在我们的实现中采用了权值共享
  - 图3-(a)运动估计模块的两个卷积之间是否有激活层，在我们的实现中加入了ReLU激活函数
  - 图5-(b)中patch大小p没有给出，在我们的实现中采用了论文"Non-Local neural networks"的nonlocal模块结构
  - 图1中，对于可用参考帧数量小于3的情况下是否使用multi-frame fusion没有说明，在我们的实现中跳过multi-frame fusion