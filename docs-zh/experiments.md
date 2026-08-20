# 实验

ICA Lens 提供两个论文级实验。重构是检验字典质量的主要实验；稀疏探测则检验少量
坐标是否集中承载了与类别有关的信息。

| 实验 | 研究问题 | 主要结果 |
| --- | --- | --- |
| 重构 | 少量可复用方向能否准确重构未见过的激活？ | Top-k 重构误差和余弦相似度 |
| 稀疏探测 | 少量分量坐标是否集中承载了与概念有关的信息？ | 平均探针准确率随所用特征数的变化 |

两个实验都可以读取本地或已发布的 Lens artifact，保存机器可读结果，支持断点续跑，
并在独立的离线步骤中生成图片。

## 重构

### 测量什么

一个有用的字典应当只用少量可复用方向，就能准确重构典型但未见过的激活。重构实验
在留出文本上检验这一点。

对于每个 token，各方法保留 top-k 方向并重构隐藏状态。ICA、已注册的公开 SAE、
PCA 和固定随机种子的随机正交基接收完全相同的留出 token。实验报告：

- **余弦相似度**：衡量重构方向与原隐藏状态方向的一致程度。
- **归一化 MSE**：还会衡量幅值误差，并以预测均值作为基线进行归一化。

当前实验研究的是 **top-k 重构**：保留最强的 k 个方向，舍弃其余方向。之后我们还会
加入互补的 **top-k 消融重构**：删除最强的 k 个方向，再用剩余方向重构。后者目前
尚未作为正式实验实现。

### 先运行冒烟测试

```bash
icalens experiment reconstruction \
  --lens sida/icalens-gpt2-small-pile10k \
  --layers 6 \
  --preset smoke \
  --baselines all \
  --output results/reconstruction-gpt2-smoke
```

完整实验使用 `--layers all --preset paper`。`paper` 预设在六个差异明显的领域中评估
1,024-token 上下文：新闻（AG News）、通用文本（WikiText）、源代码（GitHub Code）、
西班牙语 Wikipedia、中文 Wikipedia 和多轮对话（UltraChat）。这是一组紧凑的多样性
测试，而不是对所有语言分布的穷尽覆盖。

对于 GPT-2，结果还会包含一条虚线 SAE 对照曲线，只统计位置 0–63，以匹配该 SAE
训练时的上下文长度。实线 SAE 曲线和所有主要比较仍使用完整评估上下文。

### 执行与恢复

实验以数据集为最外层循环。对于每个数据集，所有待评估层通过共享的模型前向捕获并
写入检查点，再逐层评估；结果可靠写入后才删除缓存。重复命令会从已经完成的
“数据集—层”任务之后继续。

`--capture-layers-at-once` 控制内存与速度的权衡，可设为正整数或 `all`，默认为
`all`；较小的分组会重复模型前向，但降低峰值内存。诊断时可以使用
`--context-length` 和 `--max-tokens-per-dataset`。最终解析出的配置和源码来源会记录
在 `run.json` 中。

`pile10k` 预设是在 ICA 拟合语料上的分布内 sanity check，可用于诊断，但不能替代
使用留出数据的 `paper` 预设。

### 生成图片

```bash
icalens experiment figure reconstruction \
  results/reconstruction-gpt2-smoke
```

命令会在实验目录的 `figures/` 子目录中写入 PNG 和图注文本。除汇总图外，如果结果
包含多个层或多个数据集，还会生成按层和按数据集排列的子图。传入多个实验目录可以
生成对齐的模型面板；需要 PDF 时再加 `--format png,pdf`。

## 稀疏探测

### 测量什么

稀疏探测实验运行论文中的 SAEBench 协议。对于每个分类数据集，SAEBench 在训练集
上对特征维度排序，只用排名最高的少量特征训练监督线性探针，再在留出样本上报告
准确率。它衡量的是：在很小的特征预算下，与类别有关的信息有多集中。

ICA 和 PCA 坐标带有正负号，因此每个坐标会拆成正、负两个非负特征。这样，ICA、
已注册的公开 SAE、PCA 和固定随机种子的随机正交基就可以使用完全相同的数据集、
划分、特征预算和探针进行比较。

### 先运行冒烟测试

```bash
icalens experiment saebench-sparse-probing \
  --lens sida/icalens-gpt2-small-pile10k \
  --layers 6 \
  --preset smoke \
  --output results/gpt2-smoke
```

`smoke` 预设只用于兼容性检查，不是论文结果。`paper` 预设使用论文中的八个数据集、
完整训练/测试规模和标准特征预算。比较所有已注册方法：

```bash
icalens experiment saebench-sparse-probing \
  --lens sida/icalens-gpt2-small-pile10k \
  --layers 6,10 \
  --preset paper \
  --baselines all \
  --output results/gpt2-paper-comparison
```

随包发布的注册表固定了支持的公开 SAE checkpoint 及其预处理方式，包括 GPT-2 OAI
v5 ReLU、Gemma Scope JumpReLU 和 Qwen Scope TopK SAE。SAEBench 与所需 checkpoint
只在运行时按需下载。使用 `--dry-run` 可以只查看解析出的模型、后端、数据集和基线；
也可以通过 `--saebench-path /path/to/SAEBench` 使用已有 checkout。

### 执行与恢复

稀疏探测同样以数据集为最外层循环。对于一个数据集，程序通过共享的模型前向捕获所有
指定层，然后依次评估 ICA 和所有指定基线，再进入下一个数据集。只有在对应结果可靠
写入后，激活缓存才会删除。使用相同命令重新运行时，会跳过已经完成的工作。

开始前，ICA Lens 会估算最大的激活缓存，在此基础上增加 20% 安全余量，并检查输出
文件系统。只有明确愿意承担风险时才应使用 `--allow-low-disk` 绕过检查。终端面板会
显示总体进度、数据集和方法序号、已用时间与 ETA；完整 SAEBench 输出仍保存在实验
日志中。

### 生成图片

生成图片是离线步骤，不会加载模型或 SAEBench：

```bash
icalens experiment figure sparse-probing results/gpt2-smoke
```

默认生成 PNG，写入 `results/gpt2-smoke/figures/`，并附带图注文本。使用
`--format png,pdf` 可同时生成两种格式，`--force` 可覆盖已有文件；图片确认需要纳入
仓库时，再使用 `--output figures`。传入多个结果目录可以生成对齐的模型比较面板。

## 可复现性说明

- 论文实验应在干净的 Git worktree 上运行；源码存在未提交改动时，ICA Lens 会给出
  警告。
- 每种不同配置使用独立输出目录。只有配置完全一致时，已有 `run.json` 才会被复用。
- 实验数据放在 `results/`；只有最终选定的图片才发布到仓库顶层的 `figures/`。
