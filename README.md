# ELM Train

EEG → 冻结的两路 Encoder → 可训练 Projector → 80 个 soft tokens → 冻结的原始 Qwen3-4B → 类别、统计值和报告。

**运行 Agent 从本页顺序执行。** 先准备环境、数据和权重，再运行收到的实验配置。工作目录和 GPU 使用对方服务器实际分配的值，无需复刻我们的绝对路径。

当前已实现 EEG Projector 训练、恢复、vLLM 生成和报告；新的大批次 Oracle 数据生成/训练入口仍在准备。下面的35步启动小跑用于检查新服务器入口。

## 1. 一次准备环境、数据和权重

平台：Linux x86_64、NVIDIA GPU，驱动需兼容所选 CUDA 构建。建议至少200 GiB可用空间。模型执行只在授权服务器进行。

```bash
git clone https://github.com/wythedee/elm-train.git
cd elm-train

# 替换为获准使用的服务器工作目录，之后保持同一个值。
export ELM_WORK=/work/elm
bash tools/setup_server.sh
```

脚本并行安装三个独立环境，再并行下载数据、原始4B和组件权重。普通包走阿里云镜像，CUDA构建和指定wheel走官方源：

| 环节 | 环境路径 | 固定技术栈 |
|---|---|---|
| 特征提取、可微训练 | `$ELM_WORK/envs/train` | Python3.10、PyTorch2.5.1/cu124、Transformers4.57.6 |
| **独立生成、推理和评测** | `$ELM_WORK/envs/ar-v1` | **Python3.11、vLLM0.32.0+cu129 / V1**、PyTorch2.13.0/cu129、Transformers5.19.0 |
| 下载 | `$ELM_WORK/envs/download` | Python3.11、ModelScope SDK1.37.1 |

训练 forward/backward 用 PyTorch＋Transformers，让梯度经过冻结4B回传到tokens/Projector。**vLLM负责生成和推理，不负责训练梯度。** 历史Oracle的训练内Student rollout使用Transformers `generate`。三套环境不能混装；新配置使用 `ar_backend: "v1"`，不带旧的 `VLLM_USE_V1=0`。

需要Python3.10和3.11。已有其它位置的解释器时，启动前指定：

```bash
export ELM_PYTHON_TRAIN=/actual/python3.10
export ELM_PYTHON_AR=/actual/python3.11
bash tools/setup_server.sh
```

若服务器用Conda且缺少这两个版本，可在获准工作区创建：

```bash
conda create -y -p "$ELM_WORK/python310" python=3.10
conda create -y -p "$ELM_WORK/python311" python=3.11
export ELM_PYTHON_TRAIN="$ELM_WORK/python310/bin/python"
export ELM_PYTHON_AR="$ELM_WORK/python311/bin/python"
bash tools/setup_server.sh
```

缓存、临时文件、环境和日志均在 `ELM_WORK`。日志在 `setup/{train,ar-v1,download,assets}.log`；成功后有 `setup/assets_ready.json`。脚本核对模型版本、数据文件大小与条数，不再完整扫描79GB数据SHA，不运行优化步。数据上传未完成时自动等待完整发布标记，默认最长6小时。失败后保留已下载文件，按日志修复后重试；不要自动改系统驱动。

## 2. 数据和权重地址

| 资产 | 获取位置 | 使用方式 |
|---|---|---|
| 完整现有预处理数据 | [wythedee/elm-data](https://modelscope.cn/datasets/wythedee/elm-data)，`releases/native200_20261101/` | 16来源、447545段；约79.3GB信号文件，另有清单；**上传进行中** |
| 原始Qwen3-4B | [Qwen/Qwen3-4B](https://modelscope.cn/models/Qwen/Qwen3-4B) | 冻结语言模型，使用记录中的原始版本 |
| 两路Encoder | [wythedee/elm](https://modelscope.cn/models/wythedee/elm)，`releases/20261031/encoders/` | 冻结提取类别/统计特征，约334MB |
| Oracle10200 | 同一模型库的 `oracle/oracle10200.safetensors` | 冻结的80-token对齐目标及Oracle对照，约12MB |
| Projector6600 | 同一模型库的 `projector/projector_report6600.safetensors` | 单独评测已有模型，约1.01GB；fresh训练默认新初始化 |

公开下载无需令牌。代码放GitHub，数据放ModelScope dataset，权重放ModelScope model。EEG、缓存、凭据、逐人预测和大checkpoint不提交Git。

```text
$ELM_WORK/
  envs/{train,ar-v1,download}/
  data/releases/native200_20261101/
    manifests/DATASET/{train,val,test}.jsonl
    native200/DATASET/processed_average/data.mdb
    h5/MI_10_HeBin2021_LR.h5
    dataset_manifest.json
  models/Qwen3-4B/
  models/elm/releases/20261031/
  configs/    features/    runs/    setup/    cache/    tmp/
```

清单使用相对路径，配置填 `manifest_root` 和 `data_root` 即可；标签顺序、统计阈值和通道映射随权重提供。已有预处理数据可直接使用，原始数据重建见[预处理代码和来源](docs/DATASETS.md#preprocessing-code-and-source-access)。手动获取见[数据指南](docs/TRAINING_DATA.md)和[权重指南](docs/CHECKPOINTS.md)。

当前预处理池：Train313626 / Val61474 / Test72445，约577小时保留窗口。原预处理含抽样上限；共享目录2.77TB原始数据还有扩充空间。已知Mumtaz/NEMAR同源、CHB01/21同人，正式泛化需要统一核对切分和冻结Encoder来源。

## 3. 首次H20做一次短启动，然后执行实验配置

代码已在A100跑通特征、训练、真实生成和报告。首次H20执行保留一次短启动检查，无需反复跑全部静态测试、旧实验或恢复演练。

```bash
export ELM_GPU=2  # 替换成明确分配给本任务的物理GPU
python3 tools/make_server_config.py --work "$ELM_WORK" --gpu "$ELM_GPU"
export ELM_CFG="$ELM_WORK/configs/pilot35.json"
export ELM_TRAIN_PY="$ELM_WORK/envs/train/bin/python"

"$ELM_TRAIN_PY" -B -m elm_pipeline.cli --config "$ELM_CFG" check
"$ELM_TRAIN_PY" -B -m elm_pipeline.cli --config "$ELM_CFG" features
"$ELM_TRAIN_PY" -B -m elm_pipeline.cli --config "$ELM_CFG" train
"$ELM_TRAIN_PY" -B -m elm_pipeline.cli --config "$ELM_CFG" evaluate --tag pilot35
CUDA_VISIBLE_DEVICES='' "$ELM_TRAIN_PY" -B -m elm_pipeline.cli --config "$ELM_CFG" report
```

生成配置取16来源每Train/Val各2条，fresh Projector35步；调度器按阶段选择训练/vLLM环境。它检查启动与完整链路，不以35步准确率判断模型好坏。必要的冻结状态、有限Loss、checkpoint和输出覆盖检查保留。历史身份重叠仅在明确标记的软件短测里记录容许，不据此声称独立泛化。

之后按提供的正式实验配置执行相同五阶段：`check → features → train → evaluate → report`。同版本特征只提取一次，后续Loss组复用；严格恢复保持同一配置和已提交状态。已有6600权重的单独评测是可选项：

```bash
python3 tools/make_server_config.py --work "$ELM_WORK" --gpu "$ELM_GPU" \
  --mode report6600 --name report6600 --features "$ELM_WORK/features/pilot35"
# 对 configs/report6600.json 只执行 check / evaluate / report。
```

新大批次Oracle题库/训练器尚未全部交付，由我们补齐。Agent执行收到的配置，不猜缺失实现或自行设计实验。当前新入口支持类别、六统计和完整报告；其它历史推理任务单独见源码和状态说明。

## 4. 并行和效率

- 已批准且就绪的组与seed尽量使用多张获准GPU并行，一任务一卡；并发数取就绪任务数与获准显存达标GPU数的较小值。
- 每个任务独立配置、输出、日志和checkpoint；完成的特征与权重只读共享，避免重复下载和提特征。
- `gpus: [2,3]`表示候选卡，不是DDP；独立任务各填单卡，不用`torchrun`复制同一任务。
- 同一任务训练、恢复、评测顺序执行。推理用vLLM批量生成；后端细节见[版本说明](docs/VLLM_UPGRADE.md)。
- 保留显存/磁盘保护和owned进程清理，仅使用分配给本任务的资源。

## 5. 交回结果

交回 `final_report.md`、`results_summary.json`、实际配置、代码commit、数据覆盖、训练日志和失败原因。逐样本raw与checkpoint保留在私有工作区，按用户要求交回。类别、六统计、七项报告全对、格式和支持数分开列；格式正确不等于事实正确。

当前EEG训练冻结两Encoder、4B和Oracle，只训约2.54亿参数Projector，使用答案CE＋归一化token MSE＋0.1 cosine；默认不加KL、辅助头或diversity loss。Oracle阶段则训练token bank，文字Teacher用于监督/蒸馏。

深入定位再看：[完整运行参考](docs/AGENT_RUNBOOK.md)、[数据与原始入口](docs/DATASETS.md)、[实现状态](docs/PIPELINE_STATUS.md)、[实验提案](docs/NEXT_DAY_PLAN.md)、[历史Oracle](docs/ORACLE_ARCHIVE.md)。提案不替代可执行正式配置。
