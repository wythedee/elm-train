# ELM Train

把 EEG 编码器的输出转换成冻结 Qwen3-4B 能读取的 80 个 soft tokens，训练类别读取、统计读取和完整报告生成。

当前新入口是 `python -m elm_pipeline.cli`：读取 Native200 → 两路冻结 Encoder → 特征分片 → Projector SFT＋Oracle 对齐 → 自回归生成 → 独立评分与报告。Encoder、4B 和 10200 步 Oracle 保持冻结，只更新约 2.54 亿参数的 Projector。

**已通过 A100 上的小规模软件验收：**16 个来源的特征重建、30 步训练与恢复到 35 步、真实生成及报告；ADFTD 也已从原始文件接通至报告。全量训练、全部原始数据入口和 H20 尚未验收，具体边界见状态表。

## 从这里开始

- [给运行 Agent 的完整指南](docs/AGENT_RUNBOOK.md)：环境、资产、原始/预处理数据入口、路径映射、训练、恢复、生成、报告和失败处理。
- [数据与流程验收状态](docs/PIPELINE_STATUS.md)：哪些已测、哪些仍有缺口。
- [16 个来源的数据清单](docs/DATASETS.md)：条数、原始数据入口、标签与预处理版本差异、逐步校验。
- [下一步实验及交付计划](docs/NEXT_DAY_PLAN.md)：现有结果、数据身份修正、全量覆盖和两项 Loss 的消融。
- [配置模板](configs/pipeline_debug.example.json)：先修改路径和被分配的 GPU，再启动。
- [历史 Oracle 单独复现](docs/ORACLE_ARCHIVE.md)：原 3600 步编码表实验，和当前 EEG 训练入口分开。

## 技术栈：训练和推理分开环境

运行平台为 Linux、Python 3.10、NVIDIA GPU，底座使用原始 Qwen3-4B。

| 环节 | 已验收的技术栈 | 用途 |
|---|---|---|
| 特征提取与可微训练 | PyTorch 2.5.1 / CUDA 12.4、Transformers 4.57.6、NumPy 1.26.4 | 冻结 Encoder/4B，按阶段更新 Oracle bank 或 Projector；保留经过 4B 输入 embedding 的梯度 |
| 独立自回归生成与评测 | **vLLM 0.9.2**、PyTorch 2.7.0 / CUDA 12.6、Transformers 4.53.3、NumPy 2.2.6 | BF16、`prompt_embeds` soft-prefix 输入、批量生成；当前使用 V0，进程内设置 `VLLM_USE_V1=0` |
| 数据处理 | MNE、SciPy、LMDB、h5py | 原始 EEG 预处理、Native200 读取、显式路径映射 |
| 权重与评测结果 | safetensors、PyTorch checkpoint、JSON/JSONL | 推理权重、严格恢复所需优化器/RNG、原始生成和独立解析评分 |

版本分别固定在 [requirements-train.txt](requirements-train.txt) 和
[requirements-ar.txt](requirements-ar.txt)，安装命令见[运行指南](docs/AGENT_RUNBOOK.md#environment)。
**不要把两份依赖混装在同一环境。** 这是现有 A100 验收组合，H20 仍需先短测。

训练 forward/backward 使用 PyTorch＋Transformers；vLLM 用于独立生成与评测，不承担
prefix 梯度训练。历史 Oracle 训练内部的 Student rollout 使用 Transformers `generate`，
也不要与独立 vLLM 评测混淆。

## 代码入口与就绪范围

| 代码 | 当前状态 |
|---|---|
| `elm_pipeline/cli.py` | 已实现 `check / features / train / evaluate / report` 分阶段入口 |
| `elm_pipeline/encoders.py`、`features.py` | 两路冻结 Encoder 与特征重建，已做有限样本服务器核对 |
| `elm_pipeline/training.py`、`projector.py` | EEG Projector 的 SFT＋token 对齐、保存/严格恢复，服务器 30→35 步短测通过 |
| `elm_pipeline/evaluation.py` | 实际 vLLM 生成、完整题目覆盖检查、独立评分和报告，已跑通 |
| `elm_pipeline/adapters/` | 已提供九个原始数据入口；目前仅 ADFTD 完成新 raw 输入验收，逐项状态见数据清单 |
| `tools/reproduce.py` | 已有历史 3600 步 Oracle 的隔离复现入口，需要另外提供其数据和模型资产 |
| `reference/` | 历史源码快照，不能直接当作新批次运行命令 |
| 新 10 万候选题／3 万步 Oracle 批次 | 目前为提案，新题库构建、变体训练和批量 launcher 尚未接齐；不是 clone 后就能启动的六组实验 |

## 给运行 Agent：尽量多卡并行

**资源允许时，把已经批准的实验组和随机种子分配到尽可能多的 GPU 同时运行，不要默认逐组串行。** 并发数取“已批准且就绪的任务数”和“实际获准使用、显存达标的 GPU 数”中的较小值，不默认限制成 1、2 或 4 卡。

- 当前入口是一任务一卡；每个任务分配不同物理卡，在其配置中写单卡 `gpus: [编号]`。`gpus` 填多个编号只表示候选卡，不会把一个训练任务变成多卡训练；不要直接用 `torchrun` 启动多份重复训练。
- 每个任务使用独立配置、输出目录、日志、checkpoint、缓存和锁；模型权重可以只读共享。同一任务的训练、恢复和评估按顺序执行，不并发写同一目录。
- 先在一张 H20 做同一实现的短测和速度标定，通过后并行展开；空出卡就接下一个已批准任务。按可用显存和分配白名单调度，保留显存/磁盘保护，不终止他人进程。
- Oracle 下一批建议扩大到 **10 万候选训练问题、每组 3 万训练步**，在 1/2/3 万步统一评估。这里指增加问题量和优化步数，**学习率不跟着调大**。详细配置见[计划](docs/NEXT_DAY_PLAN.md#h20-oracle-扩大训练提案)。

以上 Oracle 数字是新一轮提案，尚未启动；新批量训练入口及 H20 验收仍需补齐。不要把历史 `configs/next_batch.json` 的 3600 步复现当成新批次，也不要修改已关闭运行来增加预算。

## 先做静态检查

```bash
git clone git@github.com:wythedee/elm-train.git
cd elm-train
python3 -B tools/test_pipeline_static.py
python3 -B tools/test_validate_data.py
python3 -B tools/workbench.py check
python3 -B tools/test_reproduce.py
```

这些命令不训练、不加载模型、不使用 GPU。模型执行只在授权 Linux 服务器进行。

## 数据与模型

原始 EEG、缓存、权重和环境不在 Git 中。4B 使用官方原始 Qwen3-4B；项目模型资产单独提供，未发布的文件须等待清单，不能自动换成其他模型。数据按照配置中的根目录和路径映射读取，不要求复刻原服务器的目录。

校验优先使用条数、标签、通道/采样率/单位、窗口、受试者身份和坏样本记录，不强制所有原始文件 SHA 相同。历史清单为 Train313626 / Val61474 / Test72445，包含窗口限制及同源名称，不能直接解释为独立全量原始记录。

**已知切分问题：**MUMTAZ 与 NEMAR_MDD 同源，CHB01 与 CHB21 也是同一人。历史独立切分有重叠，当前结果用于 debug；正式泛化必须同时检查数据切分和冻结 Encoder 的训练来源。

## 当前训练设置

每条 EEG 输出 80×2560：32 个类别位置和六组各 8 个统计位置。两路特征保留 200→50 维压缩及每通道两个时间位置。监督为答案 CE＋归一化 token MSE＋0.1 cosine 对齐；正确类别和统计值仅用于监督，不作为 EEG 学生的文字输入。默认不加 KL、辅助头或 diversity loss。

报告分开列类别、六项统计、完整报告七项全对、格式有效率和支持数。旧报告训练的 6600 步 Val 全对为 8/128，格式为 100%；这两者不能混用。详情与限制见下一步计划。
