# 拟合与发布

ICA Lens 安装后提供了命令行工具，可以从文本或对话拟合 Lens，并发布生成的产物。
安装命令如下：

```bash
pip install icalens
```

目前，端到端拟合命令需要 CUDA GPU。

## 从文本拟合

可以先运行一个小规模示例，检查完整的拟合流程：

```bash
icalens fit text \
  --model openai-community/gpt2 \
  --dataset NeelNanda/pile-10k \
  --split train \
  --text-field text \
  --layers 6 \
  --token-budget 1000 \
  --max-iter 20 \
  --output icalens-output/quick-test
```

![在终端中使用 Pile-10k 拟合 GPT-2 ICA Lens](https://icalens.readthedocs.io/en/latest/assets/fit.png){ loading=lazy }

*使用安装后的 `icalens` 命令完成一次单层拟合。图中的运行耗时约 10 秒；实际耗时取决于
硬件，以及模型和数据是否已经缓存。*

在这个具体设置中：

- `--model openai-community/gpt2` 选择 GPT-2 Small checkpoint。它的隐藏宽度为 768，
  因此满宽 Lens 包含 768 个 ICA 成分。
- `--dataset NeelNanda/pile-10k --split train --text-field text` 从该数据集训练 split 的
  `text` 列读取纯文本。
- `--layers 6` 只捕获 transformer block 6 之后的残差流。
- `--token-budget 1000` 使用 1,000 个 token 激活进行拟合。这个规模刻意设置得很小，
  但仍超过中心化后拟合 768 个成分所需的最少 769 个 token。
- `--max-iter 20` 固定执行 20 次 FastICA 更新。在这个小 token budget 下，图中机器
  完成整个示例约需 10 秒。
- `--output icalens-output/quick-test` 将 manifest 和拟合层保存到该目录。

要拟合一个更具实用性的 GPT-2 Lens，可以增加 token budget 和迭代次数：

```bash
icalens fit text \
  --model openai-community/gpt2 \
  --dataset NeelNanda/pile-10k \
  --split train \
  --text-field text \
  --layers all \
  --capture-layers-at-once 1 \
  --token-budget 1000000 \
  --max-iter 20 \
  --output icalens-output/icalens-gpt2-small
```

这两个命令都会解析并记录模型与数据集的精确 revision。程序会对指定文本字段进行
tokenize，捕获每个 block 之后的残差流激活，拟合指定层，并在每一层完成后立即保存
checkpoint。

使用 `--token-budget all` 可以利用所选数据集中的全部可用 token 进行拟合。tokenize
完成后，命令会输出最终解析得到的 token 数量。

## 从对话拟合

使用 UltraChat 对话为指令微调的 Qwen3.5 拟合 Lens：

```bash
icalens fit chat \
  --model Qwen/Qwen3.5-2B \
  --dataset HuggingFaceH4/ultrachat_200k \
  --split train_sft \
  --layers all \
  --capture-layers-at-once 1 \
  --token-scope all \
  --token-budget 1000000 \
  --max-iter 20 \
  --output icalens-output/icalens-qwen3.5-2b-ultrachat-1m
```

捕获激活前，程序会使用模型 tokenizer 的聊天模板渲染对话。数据集必须提供由 `role`
和 `content` 字段组成的消息列表；如果消息位于其他列，可以用 `--messages-field` 指定。

`--token-scope all` 允许从渲染后的所有位置采样，包括角色标记和模板 token。其他可选
范围是 `content`、`user` 和 `assistant`。选择更窄的范围只会改变被采样的激活行；
渲染后的完整对话仍然是模型使用的上下文。

## 重要参数

| 参数 | 作用 |
| --- | --- |
| `--model` | Hugging Face 语言模型仓库 |
| `--dataset` | Hugging Face 拟合数据集仓库 |
| `--split` | 要流式读取的数据集 split |
| `--text-field` | `icalens fit text` 使用的纯文本列 |
| `--context-length` | 每篇文档最多保留的 token 数 |
| `--layers` | 逗号分隔、从 0 开始的 transformer block 编号，或 `all` |
| `--token-budget` | 拟合使用的采样激活行数 |
| `--candidate-tokens` | 候选 token 池大小；默认等于 token budget |
| `--capture-layers-at-once` | 同时在 CPU 内存中保存的层数；`0` 表示全部指定层 |
| `--fit-batch-size` | 每次在 GPU 上处理的激活行数；`0` 表示一次放入全部拟合数据 |
| `--max-iter` | 固定的 FastICA 迭代次数 |
| `--objective-every` | 每 N 次迭代记录一次目标函数分位数 |
| `--seed` | token 采样和 FastICA 初始化的随机种子 |
| `--max-vram-gb` | 用于测试显存预算的可选 PyTorch CUDA allocator 上限 |

FastICA 会执行指定的固定迭代次数，不会根据 tolerance 提前停止。由于中心化会损失一个
秩，满宽 ICA 还要求至少有 `hidden_size + 1` 个采样激活。

## 内存行为

捕获的激活保存在 CPU 内存中，拟合时则在 GPU 上分批处理。两个主要参数分别控制不同
的资源：

- 减小 `--capture-layers-at-once` 可以降低 CPU 激活内存占用。设为 `1` 时，程序会
  捕获并拟合一层，然后再处理下一层。
- 减小 `--fit-batch-size` 可以降低拟合阶段的 GPU 内存占用。

模型前向传播会在当前捕获组所需的最深层结束。每一层拟合完成后都会立即保存，包括其
目标函数历史。因此，即使后续层的处理被中断，之前完成的层仍然可以使用。

## 拟合已有激活

如果已经有激活张量，可以使用 Python API。张量最后一维必须等于模型隐藏维度；所有
前导维度都视为样本维度。下面的完整示例使用符合 GPT-2 隐藏宽度的合成激活，因此可以
直接运行。要得到有意义的 Lens，请将其替换为从实际数据捕获的激活。

```python
import torch

from icalens import ICALens

# A full-width 768-component fit needs at least 769 samples after centering.
generator = torch.Generator().manual_seed(0)
activations = torch.randn(1000, 768, generator=generator)

lens = ICALens(
    model_id="openai-community/gpt2",
    model_type="base",
    activation_site="resid_post",
)

lens.fit(
    activations,
    layer=6,
    max_iter=20,
    batch_size=8192,
    device="cuda",
    progress=True,
    provenance={
        "dataset": {"description": "synthetic API example"},
        "fitting_tokens": int(activations.shape[0]),
    },
)

output = lens.save("icalens-output/my-icalens")
print(f"Saved layer {lens.available_layers} to {output}")
```

再次调用 `fit()` 会在同一个 Lens 中添加或替换一层。传入 `n_components` 可以拟合少于
隐藏维度的成分。

## 保存了什么

ICA Lens 产物会记录解释和复用该变换所需的信息：

- 待分析模型的 ID、类型和精确 revision；
- 激活位置和层编号约定；
- L2 预处理和拟合中心；
- 读取矩阵与写入矩阵；
- FastICA 配置、目标函数历史和成分排序；
- 可用层和各层成分数量；
- 拟合时提供的数据集与采样来源信息。

产物不包含待分析语言模型的权重。

## 拟合后为成分建立画像

成分画像不会重新运行 FastICA。它将标注数据集以流式方式送入模型，并为每个成分
记录正负分数比例、两侧的平方分数能量比例与主导符号、高能量 token 样例及计数，
以及写入方向经过最终归一化层和反嵌入后的高低排名词元。

```bash
icalens profile \
  --lens icalens-output/icalens-qwen3.5-2b-ultrachat-1m \
  --layers all \
  --dataset HuggingFaceH4/ultrachat_200k \
  --split train_sft \
  --token-scope all \
  --max-tokens 100000 \
  --top-k-examples 20 \
  --min-energy 0.05 \
  --output icalens-output/icalens-qwen3.5-2b-profiled
```

每完成一层都会写入输出目录。画像数据集及其精确 revision 与拟合来源分开记录。
画像以可选 JSON 文件保存在
`component_profiles/` 下，不会改变已经拟合的中心、读取矩阵或写入矩阵。因此，已有
的本地或已发布 Lens 可以在之后补充画像，无须重新拟合。

```python
profile = lens.component_profile(layer=5, component=188)
print(profile["dominant_sign"])
print(profile["examples"]["negative"]["tokens"])
print(profile["logit_lens"]["dominant"]["top_tokens"])
```

Logit Lens 结果只是诊断性关联：对于中间层，它跳过了后续 Transformer 块，不能视为
对生成 token 的精确因果预测。

## 使用 Hugging Face 身份认证

ICA Lens 产物应存放在 Hugging Face **Model** 仓库中。使用标准 Hugging Face CLI 和
具有写权限的 token 登录：

```bash
hf auth login
```

也可以在环境中设置 `HF_TOKEN`。发布命令还会读取当前目录 `.env` 文件中的
`HF_TOKEN`：

```dotenv
HF_TOKEN=hf_...
```

不要提交该文件或 token。

## 发布并验证

发布保存好的产物：

```bash
icalens publish \
  --lens icalens-output/icalens-gpt2-small \
  username/icalens-gpt2-small
```

添加 `--private` 可以创建私有仓库。命令会上传产物，重新下载其 manifest，并验证远端
元数据和可用层是否与本地 Lens 一致。

对应的 Python 写法是：

```python
lens.push_to_hub("username/icalens-gpt2-small")
```

发布后，可以在任意环境中这样加载：

```python
from icalens import ICALens

lens = ICALens.from_pretrained("username/icalens-gpt2-small")
```
