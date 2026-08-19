# 实验

ICA Lens 可以从本地或已发布的 Lens artifact 运行论文中的 SAEBench
稀疏探测协议。安装 ICA Lens 时不会下载 SAEBench；首次运行实验时，命令会根据
模型选择兼容的仓库与固定 commit，并在托管缓存中准备独立环境。

## GPT-2 冒烟测试

先用单层和小规模数据检查完整流程：

```bash
icalens experiment saebench-sparse-probing \
  --lens sida/icalens-gpt2-small-pile10k \
  --layers 6 \
  --preset smoke \
  --output results/gpt2-smoke
```

`smoke` 预设用于兼容性检查，并不是论文结果。`--preset paper` 才会使用论文的
八个数据集、训练/测试规模和特征数量。稀疏探测适配器把每个带符号 ICA 分数拆成
正、负两个非负特征，使探针可以分别选择两个方向。每个完成的层都会立即记录；用
相同输出目录重新执行时会复用已有结果。

## 与 SAE、PCA 和随机基比较

GPT-2、Gemma 2 2B 和 Qwen3.5 9B Base 已注册公开 SAE 与 PCA 基线。所有模型还
可以使用固定随机种子的满秩随机正交基。下面的命令让四种方法使用完全相同的数据集、
划分、特征预算和 SAEBench 探针：

```bash
icalens experiment saebench-sparse-probing \
  --lens sida/icalens-gpt2-small-pile10k \
  --layers 6,10 \
  --preset paper \
  --baselines sae,pca,random \
  --output results/gpt2-paper-comparison
```

注册表中的 SAE 分别使用 GPT-2 OAI v5 ReLU、Gemma Scope JumpReLU 和 Qwen
Scope TopK-50 checkpoint。Qwen3.5 9B SAE 的宽度为 65,536，每个选中的层需要下载
约 2.15 GB。PCA 与 ICA Lens 使用相同的 L2 归一化和中心化拟合空间；由于该 Lens
是满秩的，可以直接从 artifact 中保存的白化变换恢复拟合时的 PCA 基，无须再次
采集拟合语料。ICA 和 PCA 的带符号坐标都会拆成正、负两个特征。

在每一层内，SAEBench 只采集一次原始模型激活，并临时保存，供 ICA、SAE、PCA 和
Random 共用；所有指定表示完成后，缓存会自动删除。断点续跑以 `(层, 方法)` 为单位，
因此后来增加基线时只会补跑缺少的方法。

各数据集依次运行：先采集一个数据集，完成所有方法的评估并删除缓存，再开始下一个。
ICA Lens 会估算最大单个数据集的缓存，并与输出目录所在文件系统的可用空间比较。检查
包含 20% 的安全余量；空间不足时会提前停止。只有在明确愿意承担风险时才应使用
`--allow-low-disk` 绕过检查。使用 float32 的 paper 预设时，缓存峰值约为：GPT-2
Small 4.6 GiB、Gemma 2 2B 13.7 GiB、Qwen3.5 9B Base 24.4 GiB。

评估期间，终端会保持一个紧凑的实时面板，显示数据集—方法任务的总体百分比、当前任务、
耗时和最近几条 SAEBench 消息。未经删减的完整输出保存在
`layers/layer_XX/saebench-detail.log`。

公开 checkpoint 的版本、层命名、宽度与预处理方式记录在随包发布的基线注册表中。
只有明确为当前模型注册的基线才能运行。公开 checkpoint 与 SAEBench 都只在真正
运行实验时按需下载。

如只想查看解析结果，而不下载或运行 SAEBench：

```bash
icalens experiment saebench-sparse-probing \
  --lens sida/icalens-gpt2-small-pile10k \
  --layers 6 \
  --preset smoke \
  --output results/gpt2-smoke \
  --dry-run
```

开发时可以通过 `--saebench-path /path/to/SAEBench` 使用已有 checkout。命令只接受
注册表中明确验证过的模型与后端组合。

## 生成论文图片

生成图片是离线步骤，不会加载语言模型或 SAEBench：

```bash
icalens experiment figure sparse-probing results/gpt2-smoke
```

使用 `--format png,pdf` 同时生成两种格式，使用 `--force` 覆盖已有文件。默认生成
的临时图片与实验结果放在一起，即 `results/gpt2-smoke/figures/`。比较实验会为
每种方法画一条曲线；多层结果会按特征预算取平均。

确认图片需要纳入仓库时，再显式发布到顶层 `figures/`：

```bash
icalens experiment figure sparse-probing results/gpt2-smoke --output figures
```

## 留出数据重构

重构实验检验：对于未参与拟合的激活，能否只用少量可复用的字典方向准确重构。ICA、
已注册的公开 SAE、PCA 和固定随机种子的随机正交基使用完全相同的留出 token。每个
token 都按单个方向对重构向量的贡献范数排序。主要指标是归一化 MSE，同时报告余弦
相似度作为方向性指标。

先运行 GPT-2 兼容性测试：

```bash
icalens experiment reconstruction \
  --lens sida/icalens-gpt2-small-pile10k \
  --layers 6 \
  --preset smoke \
  --baselines all \
  --output results/reconstruction-gpt2-smoke
```

完整留出数据实验使用 `--layers all --preset paper`。该预设有意选择六种差异明显的
领域：新闻（AG News）、百科文本（WikiText）、源代码（GitHub Code）、非英语文本
（西班牙语 Wikipedia 和中文 Wikipedia）以及多轮对话（UltraChat）。这是一组紧凑的多样性测试，而非
对所有语言分布的穷尽覆盖。对话统一渲染为带有 `User:` 和 `Assistant:` 标签的纯
文本，保证不同模型接收完全相同的输入。

该预设使用 1,024-token 上下文。
对于 GPT-2，结果还会额外给出一条虚线 SAE 对照曲线，只统计位置 0–63，以匹配该
SAE 训练时使用的上下文长度。实线 SAE 曲线和所有主要比较仍使用完整上下文。

执行顺序以数据集为最外层循环。对于每个数据集，ICA Lens 会在一次共享的模型前向中
捕获所有待评估层，将激活写入检查点，再逐层评估；所有结果可靠写入后才删除该数据集
的缓存。用相同命令重新运行时，会跳过已经完成的“数据集—层”任务。紧凑的终端面板
会显示当前数据集、当前层、总体进度、已用时间和预计剩余时间。
`--capture-layers-at-once` 控制内存与速度的权衡，可设为正整数或 `all`，默认为
`all`；较小的分组会重复模型前向，但能降低峰值内存。宽 SAE 字典的评估会在内部自动
分批。诊断实验可用 `--context-length` 覆盖预设值，实际使用的值会记录在
`run.json` 中。

`pile10k` 预设是在 ICA 拟合语料上进行的分布内诊断，可用于和早期重构实验比较，
但不能替代使用留出数据的 `paper` 预设。

离线生成 NMSE 与余弦相似度图片：

```bash
icalens experiment figure reconstruction \
  results/reconstruction-gpt2-smoke
```

传入多个实验目录可以生成对齐的多模型面板。默认会在第一个实验目录的 `figures/`
子目录中写入 PNG、PDF 和简短图注文本。
