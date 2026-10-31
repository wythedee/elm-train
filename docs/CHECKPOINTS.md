# 下载哪份权重，什么时候用

模型仓库：[ModelScope wythedee/elm](https://modelscope.cn/models/wythedee/elm)。
统一发布目录为 `releases/20261031/`，大小约 **1.36 GB**，不含原始 Qwen3-4B。
精确文件、大小和完整 SHA 固定在 [发布清单](../metadata/modelscope_release.json)。

| 文件（相对发布目录） | 是什么 | 什么时候用 |
|---|---|---|
| `encoders/encoder_class.safetensors`，314 MB | 类别 Encoder：16 个 CBraMod 专家、Top3 router、adapter、200→50 压缩 | 从原始/Native200 EEG 重建类别特征；已有同版本缓存可复用 |
| `encoders/encoder_statistics.safetensors`，19.6 MB | 统计 Encoder | 与类别 Encoder 一起提特征，输出另一组 `[B,1,150,50]` 特征 |
| `oracle/oracle10200.safetensors`，11.6 MB | 10200 步的 80-token Oracle bank | 冻结的 token 对齐目标、Oracle 理想输入对照；不是 EEG Encoder 或语言模型 |
| `projector/projector_report6600.safetensors`，1.01 GB | 6600 步的 EEG Projector，253616432 参数 | 直接复核现有类别/统计/报告模型，或在明确的新计划中做已有权重初始化；不是 fresh 基线 |

四份均为 safetensors，含推理/初始化所需参数，**不含优化器、RNG、分类头或原始私有 checkpoint**。
两个 Encoder 文件已包含该特征提取器所需的骨干、适配器和压缩参数，不需要再下载一个骨干权重覆盖它们。
它们不是从头训练 16 专家时所需的完整训练状态。严格续训须另外使用原完整 checkpoint。

## 1. 在服务器下载和校验

不需要令牌，也不需要安装 ModelScope SDK。下载器只用 Python 标准库；保留断点，
对已有文件先验完整 SHA，发现不同文件时保留并拒绝覆盖；每个下载根目录只允许一个下载器。

```bash
# 元数据计划；Mac 上也可以运行，不联网、不下载权重。
python3 -B tools/download_checkpoints.py --plan

# 以下只在授权 Linux 服务器执行。路径应为私有工作区，不在 Git checkout 内。
python3 -B tools/download_checkpoints.py --root /work/elm-models/elm

# 不联网，完整复核已经下载的全部文件。
python3 -B tools/download_checkpoints.py --root /work/elm-models/elm --verify-only

# 只准备 fresh Projector 训练所需的冻结 Encoder + Oracle，省去约1GB已训练Projector。
python3 -B tools/download_checkpoints.py --root /work/elm-models/elm \
  --components encoders oracle
```

`--root` 是 ModelScope 仓库的本地根目录，下载后会保留 `releases/20261031/` 层级。
默认留 20 GiB 磁盘余量。中断后执行同一命令续传，不删除其他数据，不改已声明的文件 SHA。
ModelScope 的 `master` 是定位方式，固定发布路径和 SHA 决定身份；不能静默换成另一个 checkpoint。

也可以使用下载客户端环境中的 ModelScope SDK：

```python
from modelscope import snapshot_download
snapshot_download(model_id="wythedee/elm", revision="master",
                  allow_patterns=["releases/20261031/*"],
                  local_dir="/work/elm-models/elm")
```

SDK 下载后仍可运行上面的 `--verify-only`。不要把令牌放进公开脚本、配置、命令行或 Git。

## 2. 单独下载原始 4B

使用官方 [Qwen/Qwen3-4B](https://modelscope.cn/models/Qwen/Qwen3-4B)，放在
`/work/elm-models/Qwen3-4B`。它不包含在本次 1.36 GB 内，不能替换成 2507、3.5、8B 或量化版。
下载方式及十个原始文件的核验见 [运行指南](AGENT_RUNBOOK.md#asset-inventory)。
Encoder、Oracle 和 Projector 的语义/宽度对应这一个底座。

## 3. 三种使用方式

**直接评测已有 6600 步模型**：复制 [推理配置](../configs/pipeline_report6600.example.json)，
修改数据 manifest、路径映射、输出目录和获准 GPU。先执行 `check`、`features`，再执行
`evaluate`、`report`；不要调用 `train`。该配置设置 `train_steps: 0`，不会被当成训练配置使用。

```bash
/work/envs/elm-train/bin/python -B -m elm_pipeline.cli \
  --config /work/elm-private/report6600.json check
/work/envs/elm-train/bin/python -B -m elm_pipeline.cli \
  --config /work/elm-private/report6600.json features
/work/envs/elm-train/bin/python -B -m elm_pipeline.cli \
  --config /work/elm-private/report6600.json evaluate --tag report6600
CUDA_VISIBLE_DEVICES='' /work/envs/elm-train/bin/python -B -m elm_pipeline.cli \
  --config /work/elm-private/report6600.json report
```

外层调度仍使用训练环境；`evaluate` 会按 `ar_python` 选择独立 vLLM V1 环境。
如果已有经身份核验的同版本特征，可以配置其目录并跳过 `features`，不能对已有特征目录重复写入。
示例只跑 ADFTD 的有限样本，是工程检查；实际数据集和 Val 数量必须按计划显式设置。

**训练 fresh Projector / 做 Loss 消融**：使用 [训练短测配置](../configs/pipeline_debug.example.json)，
保留相同冻结 Encoder、Oracle、schema 和 thresholds；不填 `evaluation_checkpoint` 或
`initial_projector`。先按运行指南做软件短测，再按已确认的新计划扩大。不同 Loss 使用同一
初始 Projector/随机种子和固定数据，不能一组 fresh、另一组从 report6600 开始。

**从已有权重初始化新实验**：在已确认计划中把 `initial_projector` 指向 report6600，
创建全新输出目录和优化器。文件不含优化器/RNG，所以只能叫 weight-only 初始化，不能叫
“从6600步严格续训”。`evaluation_checkpoint` 只控制评测读取哪个权重，不会初始化训练。

重新训练 Oracle 不能把 oracle10200 当作原生初始化；当前新 EEG 入口使用冻结的 Oracle10200。
历史 Oracle 复现器及另外所需题库/初始化资产见 [ORACLE_ARCHIVE](ORACLE_ARCHIVE.md)，
新大批次 Teacher/Oracle 训练器仍按计划补齐，不能将这份权重发布冒充全流程从头训练已就绪。

## 配套元数据必须一起用

- `encoders/encoder_config.json`：16 专家、Top3 router、adapter 维度和权重身份。
- `oracle/oracle_config.json`：31 标签×32槽、6字段×3档×8槽、2560维；标签顺序不能改。
- `projector/projector_config.json`：32＋6×8＝80 tokens、2560维、输出尺度0.08、已训练6600步。
- `data_schema.json`：31项 Oracle 标签顺序和六字段；真实EEG为30类，额外 schizophrenia 仅合成。
- `thresholds.json`：原 Train-only 分箱阈值。当前权重应沿用；重拟合会改变 Oracle token 语义。
- `channels.json`：75个标准位置及别名；保留200→50压缩和两路各自的两个时间位置选择。

原始 EEG、Native200、特征缓存和逐人结果没有上传。数据入口和身份核对仍见
[DATASETS](DATASETS.md)。MUMTAZ/NEMAR 同源、CHB01/21 同人及冻结 Encoder 的训练来源
会影响泛化解释；权重可下载不代表这些问题已解决。

原固定128条Val上，report6600类别66/128、六项统计534/768、完整报告8/128，格式100%。
完整报告必须七项同时正确。新版vLLM的小子集结果与数值差异见 [升级验收](VLLM_UPGRADE.md)。
这是已验证的工程起点，不是已完成全量训练、完整Test或H20验收的最终模型。
