# 简单大模型推理系统

## 一、作业阶段
> 已完成

## 二、项目阶段
[我的项目文档](./PROJECT.md)

### 1. 模型结构：Self-Attention

恭喜你，来到了本项目最为核心的部分。在开始写代码前，建议你对着课上讲的大模型结构图把每一次计算所涉及的张量形状都推导一遍，尤其是对于“多头”的理解。项目已经帮你实现了kvcache的部分和RoPE等一些算子，写这些代码其实对于大模型的学习很有帮助，但是为了不让项目过于新手不友好而省略了。

在输入经过三个矩阵乘后，我们分别得到了Q、K、V三个张量，其中Q的形状为 (seq_len, q_head×dim) ，而K、V在连接完kvcache后的形状为 (total_seq_len, k_head×dim)，其中seq_len是输入序列的长度，可以大于1，total_seq_len是输入序列和kvcache的总长度 。你应该还记得课上的内容，在Q和K进行矩阵乘后，我们希望对于seq_len中的每个token每个独立的“头”都得到一个 (seq_len, total_seq_len) 的权重矩阵。这里就出现了两个问题:

第一，Q的头数和KV的头数并不一定相等，而是满足倍数关系，一般Q头数是KV头数的整数倍；假如Q的头数是32而KV头数是8，那么每4个连续的Q头用一个KV头对应。

第二，我们需要将 (seq_len, dim) 和 (dim, total_seq_len) 的两个矩阵做矩阵乘才能得到我们想要的形状，而现在的QK都不满足这个条件；你有几种不同的选择处理这个情况，一是对矩阵进行reshape和转置（意味着拷贝），再用一个支持广播（因为你需要对“头”进行正确对应）的矩阵乘进行计算，二是将这些矩阵视为多个向量，并按照正确的对应关系手动进行索引和向量乘法，这里我推荐使用更容易理解的后一种方法。

同样的，在对权重矩阵进行完softmax后和V进行矩阵乘时也会遇到这个情况。

对于每个头，完整的Self-Attention层的计算过程如下；

``` python
x = rms_norm(residual)
Q = RoPE(x @ Q_weight.T)
K = RoPE(x @ K_weight.T)
V = x @ V_weight.T
K = cat(K_cache, K)
V = cat(V_cache, V)
### 以下是你需要实现的部分
score = Q @ K.T / sqrt(dim)
attn = softmax(score)
attn_V = attn @ V
out = attn_V @ O_weight.T
residual = out + residual
```

Self-Attention的调试是很困难的。这里推荐大家使用pytorch来辅助调试。各位可以用transformers库（使用llama模型代码）来加载模型并运行，逐层检查中间张量结果。

### 2. 功能：文本生成

请在`src/model.rs`中补充forward函数的空白部分，实现generate函数。注意在foward函数的准备阶段，我们定义了几个计算用的临时张量，这是为了在多层计算中不重复分配内存，这些临时张量会作为算子函数调用的参数，你可以根据自己的需要更改这一部分（你其实可以用比这更小的空间）。

文本生成所需的采样的算子已为你写好。你需要初始化一个会被复用的kvcache，并写一个多轮推理的循环，每一轮的输出作为下一轮的输入。你需要根据用户传的最大生成token数以及是否出现结束符来判断是否停止推理，并返回完整推理结果。

所使用的模型在`models/story`中。`src/main.rs`已经为你写好了tokenizer的编码和解码，代码完成后，可以直接执行main函数。

### 3. （可选）功能：AI对话

仿照文本生成的功能，写一个实现AI对话的chat函数，之后你可以搭建一个支持用户输入的命令行应用。你需要在多轮对话中，保存和管理用户的kvcache。

你可以使用`models/chat`中的对话模型。其对话模板如下：

``` text
"{% for message in messages %}{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}"
```

这种模板语言叫做Jinja2，在本项目中你可以不用实现任意模板的render功能，直接在代码中内置这个模板。你可以忽略system角色功能。下面是一个首轮输入的例子：

``` text
<|im_start|>system
{system_message}<|im_end|>
<|im_start|>user
{user_message}<|im_end|>
<|im_start|>assistant
```

后续每轮输入也都应该使用该模板。如果你忘记了如何使用模板生成正确的输入，请回顾课堂上讲到的内容，提示：我们的模型的基础功能是故事续写。

如果你完成了项目，请向导师展示你的成果吧！其实这个项目还有很多可以拓展的地方，比如其他数据类型的支持、多会话的支持、GPU加速等等，欢迎你继续探索。
