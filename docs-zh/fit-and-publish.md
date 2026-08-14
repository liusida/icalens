# 拟合与发布

制作一个完整的 ICA Lens 遵循同一套流程：

**拟合 → 为每个已拟合层建立画像 → 发布**

| 阶段 | 产出 |
| --- | --- |
| **拟合** | 指定层的成分方向与拟合变换 |
| **画像** | 每个已拟合层的符号统计、代表性样例和 Logit Lens token |
| **发布** | Hugging Face Model 仓库中的完整 ICA Lens 产物 |

ICA Lens 为三个阶段都提供了命令行工具。安装命令如下：

```bash
pip install icalens
```

目前，端到端拟合命令需要 CUDA GPU。

## 1. 拟合成分方向

### 快速测试文本拟合

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

### 完整文本拟合

要拟合一个更具实用性的 GPT-2 Lens，可以增加 token budget 并拟合全部层：

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

对于纯文本，`--document-framing auto` 会通过版本控制下的
`model_framing.json` 注册表按精确模型解析文档边界格式。已知模型会获得相应的文档
边界上下文 token，但该 token 不会进入拟合样本。若本地注册表中没有该模型，ICALens
会查询并缓存 GitHub 上的最新注册表；也可用 `--refresh-model-registry` 主动刷新。
未知模型会安全地停止，除非用户显式指定 `--document-framing`。最终策略、注册表哈希和
证据链接都会写入拟合 provenance。

使用 `--token-budget all` 可以利用所选数据集中的全部可用 token 进行拟合。tokenize
完成后，命令会输出最终解析得到的 token 数量。

### 从对话拟合

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

### 拟合参数

| 参数 | 作用 |
| --- | --- |
| `--model` | Hugging Face 语言模型仓库 |
| `--dataset` | Hugging Face 拟合数据集仓库 |
| `--split` | 要流式读取的数据集 split |
| `--text-field` | `icalens fit text` 使用的纯文本列 |
| `--context-length` | 每篇文档最多保留的 token 数 |
| `--document-framing` | 使用精确模型注册表自动解析（`auto`），或显式指定 `none`、`prepend-bos`、`prepend-eos` |
| `--refresh-model-registry` | 从 GitHub 获取并缓存最新的文档格式注册表 |
| `--layers` | 逗号分隔、从 0 开始的 transformer block 编号，或 `all` |
| `--token-budget` | 拟合使用的采样激活行数 |
| `--candidate-tokens` | 候选 token 池大小，或用 `all` 读取整个数据集；默认等于 token budget |
| `--capture-layers-at-once` | 同时在 CPU 内存中保存的层数；`0` 表示全部指定层 |
| `--fit-batch-size` | 每次在 GPU 上处理的激活行数；`0` 表示一次放入全部拟合数据 |
| `--max-iter` | 固定的 FastICA 迭代次数 |
| `--objective-every` | 每 N 次迭代记录一次目标函数分位数 |
| `--seed` | token 采样和 FastICA 初始化的随机种子 |
| `--max-vram-gb` | 用于测试显存预算的可选 PyTorch CUDA allocator 上限 |

FastICA 会执行指定的固定迭代次数，不会根据 tolerance 提前停止。由于中心化会损失一个
秩，满宽 ICA 还要求至少有 `hidden_size + 1` 个采样激活。

### 内存行为

捕获的激活保存在 CPU 内存中，拟合时则在 GPU 上分批处理。两个主要参数分别控制不同
的资源：

- 减小 `--capture-layers-at-once` 可以降低 CPU 激活内存占用。设为 `1` 时，程序会
  捕获并拟合一层，然后再处理下一层。
- 减小 `--fit-batch-size` 可以降低拟合阶段的 GPU 内存占用。

模型前向传播会在当前捕获组所需的最深层结束。每一层拟合完成后都会立即保存，包括其
目标函数历史。因此，即使后续层的处理被中断，之前完成的层仍然可以使用。

### 拟合已有激活

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

## 2. 为每个已拟合层建立画像

拟合确定方向，画像则为每个方向提供解释所需的证据。发布前，应使用具有代表性的语料
为每个已拟合层建立画像。画像过程不会重新运行 FastICA，也不会改变拟合中心或矩阵。
它会为每个成分记录：

- 正负 token 位置所占比例；
- 平方分数能量在正负两侧的比例，以及由此得到的主导符号；
- 高能量 token 的文本、计数、分数、位置和短上下文；
- 写入方向经过最终归一化层和反嵌入后得到的高低排名词元。

标准 CLI 会从 Hugging Face 流式读取画像数据，并直接更新现有 Lens 目录：

```bash
icalens profile \
  --lens icalens-output/icalens-qwen3.5-2b-ultrachat-1m \
  --layers all \
  --dataset HuggingFaceH4/ultrachat_200k \
  --split train_sft \
  --token-scope all \
  --max-tokens 100000 \
  --top-k-examples 20 \
  --min-energy 0.05
```

`--max-tokens` 是每一层的画像 token 预算。每完成一层，其画像都会立即写入 Lens 的
`component_profiles/` 目录，因此不需要指定第二个输出路径；即使任务中断，已完成层
也会保留下来。画像数据集及其精确 revision 与拟合来源分开记录。

### 为使用自备激活拟合的 Lens 建立画像

如果拟合激活由你自行准备，请同时保留一份可以重新遍历的原始文本或完整对话。匿名
激活张量足以拟合方向，却无法提供 token 样例及其上下文。画像语料不必包含与拟合张量
完全相同的 token，但应代表相同的数据分布。如果希望画像反映拟合分布本身，则应保留
原始数据记录，以及捕获时使用的 token 范围、上下文长度、采样随机种子、采样策略和
数据集 revision。

用于创建画像的 Python 方法是复数形式的 `profile_components()`。Base 模型传入原始
文本，Instruct 模型传入消息列表：

```python
from icalens import ICALens

lens_path = "icalens-output/my-icalens"
lens = ICALens.from_pretrained(lens_path)

# Replayable source inputs retained alongside the fitting activations.
profiling_inputs = ["First document...", "Second document..."]

for layer in lens.available_layers:
    lens.profile_components(
        profiling_inputs,
        layer=layer,
        max_tokens=100000,
        top_k_examples=20,
        min_energy=0.05,
        device="auto",
        progress=True,
    )

lens.save(lens_path)
```

对于 Instruct 模型，`profiling_inputs` 中的每一项都是一段完整对话，例如：

```python
[
    {"role": "user", "content": "Explain the result."},
    {"role": "assistant", "content": "The result shows..."},
]
```

条件允许时，应保留原始数据记录，而不是只保留解码后的 token。ICA Lens 会应用记录的
tokenizer；对于对话，还会应用模型的 chat template。画像全部附加到 Lens 后，再调用
`save()` 保存。

### 查看已保存的画像

另一个单数形式的方法 `component_profile()` 用于**读取**某个已经保存的成分画像：

```python
profile = lens.component_profile(layer=5, component=188)
print(profile["dominant_sign"])
print(profile["examples"]["negative"]["tokens"])
print(profile["logit_lens"]["dominant"]["top_tokens"])
```

Logit Lens 结果只是诊断性关联：对于中间层，它跳过了后续 Transformer 块，不能视为
对生成 token 的精确因果预测。

### 完整产物包含什么

完成拟合和画像后，产物包含：

- 待分析模型的 ID、类型和精确 revision；
- 激活位置和层编号约定；
- L2 预处理和拟合中心；
- 读取矩阵与写入矩阵；
- FastICA 配置、目标函数历史和成分排序；
- 每个已画像层的符号统计、代表性高能量样例和 Logit Lens token；
- 可用层和各层成分数量；
- 相互独立的拟合来源与画像来源信息。

产物不包含待分析语言模型的权重。

## 3. 发布

ICA Lens 产物应存放在 Hugging Face **Model** 仓库中。

### 使用 Hugging Face 身份认证

使用标准 Hugging Face CLI 和具有写权限的 token 登录：

```bash
hf auth login
```

也可以在环境中设置 `HF_TOKEN`。发布命令还会读取当前目录 `.env` 文件中的
`HF_TOKEN`：

```dotenv
HF_TOKEN=hf_...
```

不要提交该文件或 token。

### 上传并验证

发布保存好的产物：

```bash
icalens publish \
  --lens icalens-output/icalens-qwen3.5-2b-ultrachat-1m \
  username/icalens-qwen3.5-2b-ultrachat-1m
```

添加 `--private` 可以创建私有仓库。命令会上传产物，重新下载其 manifest，并验证远端
元数据和可用层是否与本地 Lens 一致。

对应的 Python 写法是：

```python
lens.push_to_hub("username/icalens-qwen3.5-2b-ultrachat-1m")
```

发布后，可以在任意环境中这样加载：

```python
from icalens import ICALens

lens = ICALens.from_pretrained("username/icalens-qwen3.5-2b-ultrachat-1m")
```
