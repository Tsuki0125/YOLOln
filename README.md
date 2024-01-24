# 基于博客园-老农的博客复现的YOLO V3

原文链接：[老农的博客-博客园](https://www.cnblogs.com/zkweb/p/14403833.html)

本项目基于老农公开代码的修改，仅用于个人学习用途！

## 工程文件说明

- archive1	数据集1
- archive2        数据集2
- data               经过预处理的数据集
- main.py         老农的YOLO代码

注：data目录在首次运行预处理代码后根据archive原始数据集生成。



老农代码的用法：

```python
#1 数据集预处理
python3 main.py prepare
#2 训练模型
python3 main.py train
#3 测试模型
python3 main.py eval
 # 然后根据终端提示输入待测试的图片路径
 # 将生成一个output图片文件
#4 输入视频文件测试
python3 main.py eval-video
```





## 关于源码改动的说明

由于现在已经是2024年了，距离原作者老农开源代码已经过去近3年，无论是python还是导入的第三方库版本都发生了一部分变化。

因此本工程的代码对比老农的博客中开源的源码有部分改动，本工程的环境如下：

- python 3.9.18
- torch 2.0.0
- numpy 1.23.5
- Pillow 10.0.1
- opencv-python 4.9.0.80