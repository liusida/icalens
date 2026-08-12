# Python API

本包的公开接口包括 `ICALens`、`CaptureResult`、`AnalysisResult` 以及异常类型。

```python
from icalens import ICALens
```

## 创建或加载 Lens

### `ICALens(...)`

```python
lens = ICALens(model_id="openai-community/gpt2", model_type="base")
```

| 参数 | 类型 / 默认值 | 含义 |
| --- | --- | --- |
| `model_id` | `str`，必填 | 待分析语言模型的 Hugging Face 仓库 |
| `model_revision` | `str \| None = None` | 语言模型的精确版本；若省略，则在首次捕获时解析 |
| `model_type` | `"base" \| "instruct" = "base"` | 基础模型或指令/聊天模型 |
| `activation_site` | `str = "resid_post"` | 激活位置名称 |
| `layer_indexing` | `str = "hidden_states"` | 产物中记录的层编号约定 |
| `row_normalize` | `bool = True` | 中心化和 ICA 前进行逐 token L2 归一化 |
| `norm_eps` | `float = 1e-12` | 归一化数值下限 |
| **返回** | `ICALens` | 尚未拟合的新 Lens |

### `ICALens.from_pretrained(...)`

```python
lens = ICALens.from_pretrained("sida/icalens-gpt2-small-pile10k")
```

| 参数 | 类型 / 默认值 | 含义 |
| --- | --- | --- |
| `repo_id_or_path` | `str \| Path`，必填 | Hub Model 仓库或本地 ICA Lens 目录 |
| `revision` | `str \| None = None` | ICA Lens 仓库的分支、tag 或 commit |
| `cache_dir` | `str \| Path \| None = None` | Hugging Face 缓存目录 |
| `token` | `str \| bool \| None = None` | Hugging Face 身份认证设置 |
| `local_files_only` | `bool = False` | 仅使用本地缓存 |
| `force_download` | `bool = False` | 强制重新下载 |
| **返回** | `ICALens` | 根据 manifest 初始化的 Lens |

从 Hub 加载时，程序最初只下载 `icalens.json`；每一层的张量文件会在首次使用该层时
按需下载。这里的 `revision` 指 ICA Lens 仓库版本，而 `model_revision` 指待分析语言
模型的版本。

## 分析模型输入

### `analyze(...)`

```python
result = lens.analyze("She deposited the check.", layer=6)
```

| 参数 | 类型 / 默认值 | 含义 |
| --- | --- | --- |
| `inputs` | `str \| list[dict[str, str]]`，必填 | 纯文本或完整 `{role, content}` 对话 |
| `layer` | `int`，必填 | 从 0 开始的 transformer block 编号 |
| `model` | `torch.nn.Module \| None = None` | 可选的外部语言模型 |
| `tokenizer` | tokenizer 或 `None` | 可选的外部 tokenizer |
| `token_scope` | `"all" \| "content" \| "user" \| "assistant" = "all"` | 对话中返回的位置 |
| `context_length` | `int \| None = None` | 最大编码长度 |
| `device` | `str \| torch.device \| None = "auto"` | 模型设备；自动模式优先 CUDA |
| **返回** | `AnalysisResult` | token、激活、ICA 分数和能量占比 |

### `generate(...)`

生成续写，并可选择钳制一个带符号的 ICA 坐标：

```python
baseline = lens.generate(messages, max_new_tokens=16)

steered = lens.generate(
    messages,
    layer=5,
    clamp=(188, -20.0),
    max_new_tokens=16,
)
```

```python
lens.generate(
    prompt,
    *,
    layer=None,
    clamp=None,
    max_new_tokens=64,
    device="auto",
    model=None,
    tokenizer=None,
    **generation_kwargs,
) -> str
```

| 参数 | 类型 / 默认值 | 含义 |
| --- | --- | --- |
| `prompt` | `str \| list[dict[str, str]]`，必填 | 原始提示词或聊天消息；消息会自动套用聊天模板 |
| `layer` | `int \| None = None` | 要修改的残差流层；使用 `clamp` 时必填 |
| `clamp` | `tuple[int, float] \| Mapping[int, float] \| None = None` | 一个 `(成分, 目标分数)`，或多个同时钳制目标的映射；应用于每个处理位置和每个生成步骤 |
| `max_new_tokens` | `int = 64` | 最多生成多少个续写 token |
| `device` | `str \| torch.device \| None = "auto"` | 模型设备；自动模式优先 CUDA |
| `model` | `torch.nn.Module \| None = None` | 可选的外部语言模型 |
| `tokenizer` | tokenizer 或 `None` | 可选的外部 tokenizer |
| `**generation_kwargs` | 关键字参数 | 继续传给 `model.generate()` 的参数 |
| **返回** | `str` | 只包含续写、不包含提示词的解码文本 |

不提供 `clamp` 时，这是普通生成。提供 `clamp` 后，ICA Lens 会修改指定层的
`resid_post` 激活，并在送回模型之前恢复每个激活的原始范数。默认使用贪心解码；也可
传入标准生成参数选择其他策略。语言模型会按需加载，并由同一个 Lens 的后续调用复用。

### `capture(...)`

```python
captured = lens.capture("She deposited the check.", layer=6)
```

| 参数 | 类型 / 默认值 | 含义 |
| --- | --- | --- |
| `inputs` | 文本或消息列表，必填 | 模型输入 |
| `layer` | `int`，必填 | 要捕获的 block |
| `model`, `tokenizer` | `None` | 可选的外部模型和 tokenizer |
| `token_scope` | `str = "all"` | 对话返回位置 |
| `context_length` | `int \| None = None` | 最大编码长度 |
| `device` | `str \| torch.device \| None = "auto"` | 模型设备 |
| **返回** | `CaptureResult` | token、位置、ID 和对齐激活 |

### `unload_model()`

| 参数 | 类型 / 默认值 | 含义 |
| --- | --- | --- |
| **返回** | `None` | 释放 `capture()`、`analyze()` 或 `generate()` 缓存的模型和 tokenizer |

此方法不会卸载 ICA 层矩阵。

## 变换激活张量

### `transform(...)`

| 参数 | 类型 / 默认值 | 含义 |
| --- | --- | --- |
| `activations` | 浮点 Tensor 或 ndarray，必填 | 最后一维等于 `lens.hidden_size` 的激活 |
| `layer` | `int`，必填 | 使用的已拟合层 |
| **返回** | 同输入数组类型 | 保留前导维度的带符号 ICA 分数 |

### `energy(...)`

| 参数 | 类型 / 默认值 | 含义 |
| --- | --- | --- |
| `scores` | 浮点 Tensor 或 ndarray，必填 | 带符号成分分数 |
| **返回** | 同输入数组类型 | 每个向量内的非负成分能量占比 |

### `keep_topk(...)`

| 参数 | 类型 / 默认值 | 含义 |
| --- | --- | --- |
| `scores` | 浮点 Tensor 或 ndarray，必填 | 成分分数 |
| `k` | `int`，必填 | 每个向量保留的最大绝对分数数量 |
| **返回** | 同输入数组类型 | 其余成分置零的副本 |

### `ablate_topk(...)`

| 参数 | 类型 / 默认值 | 含义 |
| --- | --- | --- |
| `scores` | 浮点 Tensor 或 ndarray，必填 | 成分分数 |
| `k` | `int`，必填 | 每个向量移除的最大绝对分数数量 |
| **返回** | 同输入数组类型 | 所选成分置零的副本 |

### `inverse_transform(...)`

| 参数 | 类型 / 默认值 | 含义 |
| --- | --- | --- |
| `scores` | 浮点 Tensor 或 ndarray，必填 | 最后一维等于成分数的带符号分数 |
| `layer` | `int`，必填 | 使用的 writing matrix 与中心 |
| **返回** | 同输入数组类型 | 重构的预处理激活 |

### `restore_norm(...)`

| 参数 | 类型 / 默认值 | 含义 |
| --- | --- | --- |
| `values` | 浮点 Tensor 或 ndarray，必填 | 重构或修改后的激活方向 |
| `reference` | 同类型、同形状，必填 | 用于提供每个向量目标范数的原始激活 |
| **返回** | 同输入数组类型 | 按 `reference` 恢复范数后的 `values` |

## 拟合激活张量

### `fit(...)`

```python
lens.fit(
    activations,
    layer=6,
    n_components=None,
    max_iter=20,
    random_state=0,
    progress=True,
    device="cuda",
    batch_size=8192,
    objective_every=1,
    provenance={
        "dataset": {"repo_id": "owner/dataset", "split": "train"},
        "fitting_tokens": int(activations.shape[0]),
    },
)
```

| 参数 | 类型 / 默认值 | 含义 |
| --- | --- | --- |
| `activations` | 浮点 Tensor 或 ndarray，必填 | 前导维度会展平为样本 |
| `layer` | `int`，必填 | 激活来源层的标签；不会自动捕获 |
| `n_components` | `int \| None = None` | 成分数；默认等于激活宽度 |
| `algorithm` | `str = "parallel"` | FastICA 更新算法 |
| `fun` | `str = "logcosh"` | 对比函数 |
| `max_iter` | `int = 200` | 固定迭代次数 |
| `random_state` | `int \| None = 0` | 初始化随机种子 |
| `progress` | `bool = False` | 显示进度 |
| `device` | device 或 `None` | 拟合设备 |
| `batch_size` | `int = 8192` | 每次在拟合设备处理的行数 |
| `objective_every` | `int = 1` | 每 N 次迭代记录目标分布 |
| `provenance` | `dict \| None = None` | 原样存入产物的 JSON 格式来源信息 |
| **返回** | `ICALens` | 拟合或替换该层后的同一个 Lens |

完整宽度的 ICA 至少需要 `hidden_size + 1` 个样本，因为中心化会使数据秩减少 1。

再次调用 `fit()` 会在同一个 Lens 中添加或替换一层。端到端数据集命令和完整 Python
示例见[拟合与发布](fit-and-publish.md)。

## 保存与发布

### `save(...)`

```python
path = lens.save("icalens-output/my-icalens")
```

| 参数 | 类型 / 默认值 | 含义 |
| --- | --- | --- |
| `path` | `str \| Path`，必填 | ICA Lens 产物的目标目录 |
| **返回** | `Path` | 解析后的目标目录 |

本地输出路径不会写入产物；`provenance` 中的内容会原样保存。

### `push_to_hub(...)`

```python
url = lens.push_to_hub(
    "owner/my-icalens",
    private=False,
    revision="main",
)
```

| 参数 | 类型 / 默认值 | 含义 |
| --- | --- | --- |
| `repo_id` | `str`，必填 | `owner/name` 形式的目标仓库 |
| `private` | `bool \| None = None` | 新建仓库时是否设为私有 |
| `token` | `str \| bool \| None = None` | Hugging Face 身份认证 |
| `revision` | `str = "main"` | 目标 branch 或 revision |
| `commit_message` | `str` | Hub commit 消息 |
| **返回** | `str` | Hub commit 的 URL |

## 结果对象

### `CaptureResult`

`CaptureResult` 包含 `tokens`、`token_texts`、`token_labels`、`token_tooltips`、
`token_groups`、`token_ids`、`positions` 和 `activations`。

### `AnalysisResult`

`AnalysisResult` 还包含 `scores`、`energy`、`model`、`layer`、`input_text`、
`token_scope` 和 `messages`。

在 Jupyter 或 Colab 中，将 `result` 作为单元格最后一个表达式即可显示结果：

```python
result
```

### `AnalysisResult.display(...)`

```python
result.display(metric="energy", top_k=5, height=720)
```

| 参数 | 类型 / 默认值 | 含义 |
| --- | --- | --- |
| `metric` | `"score" \| "energy" = "score"` | 成分条所显示的指标 |
| `top_k` | `int = 3` | 每张 token 卡显示的成分数 |
| `title` | `str = "ICA Lens Explorer"` | 嵌入报告标题 |
| `height` | `int = 720` | 嵌入区域的初始高度（像素） |
| **返回** | `None` | 通过 IPython 显示交互结果 |

### `AnalysisResult.to_html(...)`

```python
output = result.to_html(
    "analysis.html",
    metric="score",
    top_k=3,
    title="ICA Lens Explorer",
)
```

| 参数 | 类型 / 默认值 | 含义 |
| --- | --- | --- |
| `output_file` | `str \| Path`，必填 | 目标 HTML 文件 |
| `metric` | `"score" \| "energy" = "score"` | 成分条显示的值 |
| `top_k` | `int = 3` | 每张 token 卡显示的成分数 |
| `title` | `str = "ICA Lens Explorer"` | 报告标题 |
| **返回** | `Path` | 独立 HTML 报告路径 |

## 查看 Lens 信息

常用属性包括 `model_id`、`model_revision`、`model_type`、`activation_site`、
`hidden_size`、`available_layers` 和 `metadata`。

```python
print(lens.model_id)
print(lens.model_revision)
print(lens.model_type)
print(lens.activation_site)
print(lens.hidden_size)
print(lens.available_layers)
print(lens.metadata)
```

`metadata` 返回可移植 manifest 的独立字典副本。

## 异常

```python
from icalens import ArtifactError, ICALensError, NotFittedError
```

- `ICALensError`：本包所有异常的基类。
- `ArtifactError`：ICA Lens 产物缺失、损坏或不兼容。
- `NotFittedError`：请求的层尚未拟合或加载。
