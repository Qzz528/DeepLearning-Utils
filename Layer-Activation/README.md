### pytorch 激活函数 Activation 介绍及代码

激活函数是非线性的函数，其不改变数据的尺寸，但对输入的数据值进行变换。类似人类神经元，当输入电信号达到一定程度则会激活，激活函数对于不同大小的输入，输出值应当可体现激活和抑制的区别。

Softmax激活函数比较特殊，其形式和功能比起激活和抑制，与损失函数联系更紧密，暂不在此介绍。

以下内容源自torch官方文档，个人进行了解释补充。

https://pytorch.org/docs/stable/_modules/torch/nn/modules/activation.html

运行各种激活函数代码前请先运行如下代码，创建输入数据

```python
import torch
data = torch.arange(8)/4.-1
inputs = data.view(1,-1)
print(inputs.shape) #=>torch.Size([1, 5])
print(inputs) #=> tensor([[-1.0000, -0.5000,  0.0000,  0.5000,  1.0000]])
```

每种激活函数的代码在torch中有两种调用方法：类实例化和直接调用函数，另外附有根据公式的自写方法。

#### 常用激活函数

##### Sigmoid

$$
\text{Sigmoid}(x) = \frac{1}{1 + e^{-x}}
$$

<img src="file:///accessory/Sigmoid.png" title="" alt="Sigmoid.png" data-align="center">

- 输出范围0~1（可作为概率输出）
  
  - 输出均为正值（各方向梯度值均为正或者负，可能无法沿最优方向优化，训练效率低）
  - 函数单调，平滑（处处可导）

- 导数范围0~0.25（梯度逐层反向传播时值越来越小，可能梯度消失，训练效率低）
  
  - 两侧饱和（导数两侧极限值趋于0，导致可能的梯度消失，训练效率低）

- 运算复杂（耗时较长）

```python
#类实例化
from torch import nn
sigmoid = nn.Sigmoid() #构建层
outputs = sigmoid(inputs) #调用层
print(outputs) #=> tensor([[0.2689, 0.3775, 0.5000, 0.6225, 0.7311]])

#直接调用函数
import torch.nn.functional as F
outputs = F.sigmoid(inputs) #调用函数
print(outputs) #=> tensor([[0.2689, 0.3775, 0.5000, 0.6225, 0.7311]])

#自写方法
import torch
f_sigmoid = lambda x:1/(1+torch.exp(-x)) #定义函数
outputs = f_sigmoid(inputs) #调用函数
print(outputs) #=> tensor([[0.2689, 0.3775, 0.5000, 0.6225, 0.7311]])
```

##### ReLU

$$
\text{ReLU}(x) = \max(0, x)
$$

<img src="file:///accessory/ReLU.png" title="" alt="ReLU.png" data-align="center">

- 输出范围0~无穷，有下界无上界
  
  - 输出均为非负值（各方向梯度值均为正或者负，可能无法沿最优方向优化，训练效率低）
  - 函数单调，不平滑（x=0处不可导，需指定导数值）

- 导数值0或1
  
  - 右侧不饱和（输入x>0时，导数值为1，反向传播时可避免梯度消失）
  - 左侧饱和，导数为0（输入x<0，导数或者说梯度为0，神经元死亡，反向传播无法优化）

- 运算简单（耗时短）

```python
#类实例化
from torch import nn
relu = nn.ReLU() #构建层
outputs = relu(inputs) #调用层
print(outputs) #=> tensor([[0.0000, 0.0000, 0.0000, 0.5000, 1.0000]])

#直接调用函数
import torch.nn.functional as F
outputs = F.relu(inputs) #调用函数
print(outputs) #=> tensor([[0.0000, 0.0000, 0.0000, 0.5000, 1.0000]])

#自写方法
import torch
f_relu = lambda x:torch.max(torch.zeros(1),x) #定义函数
outputs = f_relu(inputs) #调用函数
print(outputs) #=> tensor([[0.0000, 0.0000, 0.0000, 0.5000, 1.0000]])
```

##### Tanh

$$
\text{Tanh}(x) = \frac{e^x - e^{-x}} {e^x + e^{-x}} 
$$

<img src="file:///accessory/Tanh.png" title="" alt="Tanh.png" data-align="center">

- 输出范围-1~1（Tanh实际上是对Sigmoid的平移与拉伸）
  
  - 输出值包含正值和负值（0中心化，可提升训练效率）
  - 函数单调，平滑（处处可导）

- 导数范围0~1（一定程度上缓解了梯度消失问题）
  
  - 两侧饱和（导数两侧极限值趋于0，导致可能的梯度消失，训练效率低）

- 运算复杂（耗时较长）

```python
#类实例化
from torch import nn
tanh= nn.Tanh() #构建层
outputs = tanh(inputs) #调用层
print(outputs) #=> tensor([[-0.7616, -0.4621,  0.0000,  0.4621,  0.7616]])

#直接调用函数
import torch.nn.functional as F
outputs = F.tanh(inputs) #调用函数
print(outputs) #=> tensor([[-0.7616, -0.4621,  0.0000,  0.4621,  0.7616]])

#自写方法
import torch
f_tanh = lambda x:(torch.exp(x)-torch.exp(-x))/(torch.exp(x)+torch.exp(-x)) #定义函数
outputs = f_tanh(inputs) #调用函数
print(outputs) #=> tensor([[-0.7616, -0.4621,  0.0000,  0.4621,  0.7616]])
```

##### LeakyReLU

$$
\text{LeakyReLU}(x) =
        \begin{cases}
        x, & \text{ if } x \geq 0 \\
        \text{negative\_slope} \times x, & \text{ otherwise }
        \end{cases}
$$

negative_slope默认0.01，可在代码中设置。（下图为了便于观察，negative_slope为0.1）

<img src="file:///accessory/LeakyReLU.png" title="" alt="LeakyReLU.png" data-align="center">

- 输出范围无下界无上界（对比ReLU，x<0时引入一个小的斜率，避免负值时梯度为0）
  
  - 输出值包含正值和负值（可提升训练效率）
  - 函数单调，不平滑（x=0处不可导，需指定导数值）
  - 形式接近线性（可能导致深层模型退化，影响效果）

- 导数值negative_slope或1
  
  - 右侧不饱和（输入x>0时导数值为1，反向传播时可避免梯度消失）
  - 左侧导数非0（输入x<0时也存在非0梯度，可以进行反向传播，避免了ReLU神经元死亡）

- 运算简单（耗时短）

```python
#类实例化
from torch import nn
leakyrelu = nn.LeakyReLU(negative_slope=0.01) #构建层
outputs = leakyrelu(inputs) #调用层
print(outputs) #=> tensor([[-0.0100, -0.0050,  0.0000,  0.5000,  1.0000]])

#直接调用函数
import torch.nn.functional as F
outputs = F.leaky_relu(inputs, negative_slope=0.01) #调用函数
print(outputs) #=> tensor([[-0.0100, -0.0050,  0.0000,  0.5000,  1.0000]])

#自写方法
import torch
negative_slope = 0.01
f_leakyrelu = lambda x:torch.max(torch.zeros(1),x) + negative_slope * torch.min(torch.zeros(1),x) #定义函数
outputs = f_leakyrelu(inputs) #调用函数
print(outputs) #=> tensor([[-0.0100, -0.0050,  0.0000,  0.5000,  1.0000]])
```

#### 近似ReLU的激活函数

##### SiLU(Swish)

$$
\text{SiLU}(x) = x * \text{Sigmoid}(x)
$$

<img src="file:///accessory/SiLU.png" title="" alt="SiLU.png" data-align="center">

- 输出范围有下界无上界（图形和名称上可以看出，其是Sigmoid和ReLU的结合）
  
  - 输出值包含正值和负值
  - 函数平滑，不单调（靠近0处的负值会被更多抑制）

- 导数值有界
  
  - 右侧不饱和（输入x>0时导数值为1，反向传播时可避免梯度消失）
  - 左侧饱和但导数值非0（输入x<0时也存在非0梯度，可以进行反向传播，避免了ReLU神经元死亡）

- 运算复杂（耗时长）

```python
#类实例化
from torch import nn
leakyrelu = nn.LeakyReLU(negative_slope=0.01) #构建层
outputs = leakyrelu(inputs) #调用层
print(outputs) #=> tensor([[-0.2689, -0.1888,  0.0000,  0.3112,  0.7311]])

#直接调用函数
import torch.nn.functional as F
outputs = F.leaky_relu(inputs, negative_slope=0.01) #调用函数
print(outputs) #=> tensor([[-0.2689, -0.1888,  0.0000,  0.3112,  0.7311]])
```

##### ELU

$$
\text{ELU}(x) = \begin{cases}
        x, & \text{ if } x > 0\\
        \alpha * (e^x - 1), & \text{ if } x \leq 0
        \end{cases}
$$

alpha默认为1，可在代码中设置。

<img src="file:///accessory/ELU.png" title="" alt="ELU.png" data-align="center">

- 输出范围有下界-1无上界（是对ReLU的优化，类似LeakyReLU）
  
  - 输出值包含正值和负值（可提升训练效率）
  - 函数单调，平滑（处处可导）

- 导数值有界
  
  - 右侧不饱和（输入x>0时导数值为1，反向传播时可避免梯度消失）
  - 左侧饱和但导数值非0（输入x<0时也存在非0梯度，可以进行反向传播，避免了ReLU神经元死亡）

- 运算复杂（耗时长）

```python
#类实例化
from torch import nn
elu = nn.ELU(alpha=1) #构建层
outputs = elu(inputs) #调用层
print(outputs) #=> tensor([[-0.6321, -0.3935,  0.0000,  0.5000,  1.0000]])

#直接调用函数
import torch.nn.functional as F
outputs = F.elu(inputs, alpha=1) #调用函数
print(outputs) #=> tensor([[-0.6321, -0.3935,  0.0000,  0.5000,  1.0000]])
```

##### Softplus

$$
\text{Softplus}(x) = \frac{1}{\beta} *
    \log(1 + e^{\beta x})
$$

beta默认为1，可在代码中设置。

<img src="file:///accessory/Softplus.png" title="" alt="Softplus.png" data-align="center">

- 输出范围有下界0无上界（可认为是对ReLU的平滑）
  
  - 输出均为正值（各方向梯度值均为正或者负，可能无法沿最优方向优化，训练效率低）
  - 函数单调，平滑（处处可导）

- 导数值有界0~1
  
  - 右侧不饱和（输入x>0时导数值为1，反向传播时可避免梯度消失）
  - 左侧饱和但导数值非0（输入x<0时也存在非0梯度，可以进行反向传播，避免了ReLU神经元死亡）

- 运算复杂（耗时长）

```python
#类实例化
from torch import nn
softplus = nn.Softplus(beta=1) #构建层
outputs = softplus(inputs) #调用层
print(outputs) #=> tensor([[0.3133, 0.4741, 0.6931, 0.9741, 1.3133]])

#直接调用函数
import torch.nn.functional as F
outputs = F.softplus(inputs, beta=1) #调用函数
print(outputs) #=> tensor([[0.3133, 0.4741, 0.6931, 0.9741, 1.3133]])
```

#### 硬激活函数

硬激活函数使用分段线性去逼近某些函数，牺牲平滑度来换取运算速度，其他特性与原激活函数类似。在此仅做画图展示，不进行重复的叙述。

##### HardSigmoid

$$
\text{Hardsigmoid}(x) = \begin{cases}
            0 & \text{if~} x \le -3, \\
            1 & \text{if~} x \ge +3, \\
            x / 6 + 1 / 2 & \text{otherwise}
        \end{cases}
$$

<img src="file:///accessory/Hardsigmoid.png" title="" alt="Hardsigmoid.png" data-align="center">

```python
#类实例化
from torch import nn
hardsigmoid = nn.Hardsigmoid() #构建层
outputs = hardsigmoid(inputs) #调用层
print(outputs) #=> tensor([[0.3333, 0.4167, 0.5000, 0.5833, 0.6667]])

#直接调用函数
import torch.nn.functional as F
outputs = F.hardsigmoid(inputs) #调用函数
print(outputs) #=> tensor([[0.3333, 0.4167, 0.5000, 0.5833, 0.6667]])
```

##### HardTanh

$$
\text{HardTanh}(x) = \begin{cases}
            \text{max\_val} & \text{ if } x > \text{ max\_val } \\
            \text{min\_val} & \text{ if } x < \text{ min\_val } \\
            x & \text{ otherwise } \\
        \end{cases}
$$

max_val默认为1，min_val默认为-1。

<img src="file:///accessory/Hardtanh.png" title="" alt="Hardtanh.png" data-align="center">

```python
#类实例化
from torch import nn
hardtanh = nn.Hardtanh(min_val=-1,max_val=1) #构建层
outputs = hardtanh(inputs) #调用层
print(outputs) #=> tensor([[-1.0000, -0.5000,  0.0000,  0.5000,  1.0000]])

#直接调用函数
import torch.nn.functional as F
outputs = F.hardtanh(inputs,min_val=-1,max_val=1) #调用函数
print(outputs) #=> tensor([[-1.0000, -0.5000,  0.0000,  0.5000,  1.0000]])
```

##### HardSwish

$$
\text{Hardswish}(x) = \begin{cases}
            0 & \text{if~} x \le -3, \\
            x & \text{if~} x \ge +3, \\
            x \cdot (x + 3) /6 & \text{otherwise}
        \end{cases}
$$

<img title="" src="file:///accessory/Hardswish.png" alt="Hardswish.png" data-align="center">

```python
#类实例化
from torch import nn
hardswish = nn.Hardswish() #构建层
outputs = hardswish(inputs) #调用层
print(outputs) #=> tensor([[-0.3333, -0.2083,  0.0000,  0.2917,  0.6667]])

#直接调用函数
import torch.nn.functional as F
outputs = F.hardswish(inputs) #调用函数
print(outputs) #=> tensor([[-0.3333, -0.2083,  0.0000,  0.2917,  0.6667]])
```

#### 附录

**为什么激活函数需要是非线性？**

以全连接层为例，其代表了对数据的一次线性变换，多层神经网络表示多次线性变换。根据线性代数或者矩阵运算规律，多次线性变换本质上可用单次线性变换表示。所以如果激活函数也是线性的，这会导致堆叠网络层数起不到效果，等同于单层。

**输出均为正值为什么会降低优化效率？**

以Sigmoid为例，其输出将导致的Zigzag现象。对神经网络层的参数求梯度，梯度（各方向导数）由于Sigmoid的特性均为同号，如下图所示，当优化目标位于x方向正向和y方向负向时，由于各方向导数为同号（只允许绿色区域的优化方向），由此优化过程迂回而效率低。

![Zigzag.png](accessory\Zigzag.png)

**什么是饱和？饱和为什么会造成梯度消失？**

饱和是指当x大于或小于某个界限时，函数的导数值为0（硬饱和）或极限趋于0（软饱和）。
以Sigmoid为例，其双侧饱和，当输入值较大或者较小时，该处激活函数的导数值为0或接近0。在反向传播过程中，上一个神经网络层参数的梯度由于求导的链式法则，会乘以激活函数的导数值使其接近0或者为0，导致参数的优化效率降低或无法优化。
Sigmoid更为特别的一点是，其导数值最大只有0.25，即便取到最大，在反向传播过程中仍会梯度不断减小。
