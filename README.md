# DocReaderMCP
An MCP server that can read code documents

## Pipeline
1. 给出文档链接
2. 找出所有页面 （链接名称一般和内容有关）
3. 利用LLM匹配用户prompt和具体页面，例如"jax如何实现linear layer?"，LLM负责找出某个相关页面
4. 下载这一页面，转换成markdown，结合用户prompt，给出回答

## TODO
- 查找多个链接

## 当前Demo的输出结果

### 使用了文档：
(doc) (base) crossia@crossia-macbook DocReaderMCP % python url_to_pdf.py https://flax.readthedocs.io/en/latest/ -c --depth 5 

请输入您的问题: how to write a linear layer with flax?
正在从 https://flax.readthedocs.io/en/latest/ 爬取文档链接...
找到 78 个文档链接
正在查找最相关页面...
最相关页面: https://flax.readthedocs.io/en/latest/api_reference/flax.nnx/nn/linear.html
正在处理页面并生成回答...
成功将 https://flax.readthedocs.io/en/latest/api_reference/flax.nnx/nn/linear.html 转换为PDF: temp.pdf

回答:
在 Flax 中创建线性层可以使用 `nnx.Linear` 模块。以下是基本用法示例：

```python
from flax import nnx
import jax.numpy as jnp

# 创建线性层 (输入特征3维，输出特征4维)
layer = nnx.Linear(
    in_features=3, 
    out_features=4,
    rngs=nnx.Rngs(0)  # 随机数生成器
)

# 前向计算 (输入形状为(批量大小, 输入特征))
x = jnp.ones((1, 3))
output = layer(x)
```

关键参数说明：
- `in_features`: 输入维度
- `out_features`: 输出维度 
- `use_bias`: 是否使用偏置项(默认True)
- `kernel_init`: 权重初始化方式(默认方差缩放)
- `bias_init`: 偏置初始化方式(默认零初始化)

该层会对输入的最后维度进行线性变换，输出形状为 `(..., out_features)`。
(doc) (base) crossia@crossia-macbook DocReaderMCP % python url_to_pdf.py https://flax.readthedocs.io/en/latest/ -c --depth 5 


### 没使用文档：
请输入您的问题: how to write a linear layer with flax?
正在从 https://flax.readthedocs.io/en/latest/ 爬取文档链接...
找到 78 个文档链接
正在查找最相关页面...
最相关页面: https://flax.readthedocs.io/en/latest/api_reference/flax.nnx/nn/linear.html
正在处理页面并生成回答...
成功将 https://flax.readthedocs.io/en/latest/api_reference/flax.nnx/nn/linear.html 转换为PDF: temp.pdf

回答:
### 使用Flax编写线性层的方法

在Flax中编写线性层(全连接层)可以通过以下方式实现：

### 基本实现方法

```python
from flax import linen as nn

class LinearLayer(nn.Module):
    features: int  # 定义输出特征数
    
    def setup(self):
        # 初始化权重和偏置
        self.kernel = self.param('kernel', 
                               nn.initializers.lecun_normal(),
                               (self.features,))  # 权重矩阵
        self.bias = self.param('bias',
                             nn.initializers.zeros,
                             (self.features,))    # 偏置向量
    
    def __call__(self, x):
        # 前向传播计算
        return x @ self.kernel + self.bias
```

### 更简洁的实现（推荐）

Flax提供了`nn.Dense`类，它是预实现的线性层：

```python
from flax import linen as nn

# 直接使用nn.Dense
model = nn.Dense(features=64)  # 64维输出的线性层
```

### 完整示例

```python
import jax
import jax.numpy as jnp
from flax import linen as nn

class MLP(nn.Module):
    hidden_size: int
    output_size: int
    
    def setup(self):
        self.dense1 = nn.Dense(features=self.hidden_size)
        self.dense2 = nn.Dense(features=self.output_size)
    
    def __call__(self, x):
        x = self.dense1(x)
        x = nn.relu(x)
        x = self.dense2(x)
        return x

# 初始化模型
model = MLP(hidden_size=128, output_size=10)
key = jax.random.PRNGKey(0)
params = model.init(key, jnp.ones((1, 784)))  # 假设输入是784维
```

### 关键点说明

1. `nn.Dense`会自动处理权重初始化
2. 输入维度会根据第一次调用自动推断
3. 可以自定义初始化方式，如：
   ```python
   nn.Dense(features=64, kernel_init=nn.initializers.he_normal())
   ```

Flax的线性层与PyTorch的`nn.Linear`功能类似，但遵循了JAX的函数式编程范式。