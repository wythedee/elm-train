# EEG Oracle 4B

让冻结的 Qwen3-4B 读懂 EEG 对应的 soft tokens，再把 EEG 编码器的输出对齐到这些 tokens。

本仓库复现第一阶段：**冻结原始 Qwen3-4B，只训练 80-token Oracle 编码表，训练 3600 步。** EEG 对齐代码保留作参考，不会自动启动；新消融实验也不会排队。

## 训练流程

1. 从 EEG 提取类别和六项统计值，按 Train 阈值分成 low / mid / high。
2. 类别选择 32 个可训练 tokens，六个统计档位分别选择 8 个，共 80 个。
3. 把这些 tokens 和问题一起交给冻结 4B，生成回答。
4. 用正确答案监督和明文 teacher 蒸馏更新编码表，4B 权重不变。

训练只接纳 **4B 在不看参考答案的明文输入下能答对的 Train 题**。保留完整 Val，不删除明文答错的题。EEG 波形和正确答案都不作为学生模型的文字输入。

## 先检查代码

~~~bash
git clone git@github.com:wythedee/elm-train.git
cd elm-train
python3 -B tools/workbench.py check
python3 -B tools/test_reproduce.py
~~~

这两步只需 Python 标准库，不用数据、模型或 GPU。

**数据和权重不在 Git 中。** 没有另外提供的统计清单、题库、teacher 答案和初始化文件时，只能检查代码，不能完整复现训练。所需文件的相对路径和完整 SHA 见 metadata/asset_requirements.json。

## 准备环境和数据

使用 Linux、NVIDIA GPU 和 Python 3.10。训练和自回归评测使用两个独立环境，版本来自原实验：

~~~bash
python3.10 -m venv /work/envs/elm-train
/work/envs/elm-train/bin/pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu124
/work/envs/elm-train/bin/pip install -r requirements-train.txt

python3.10 -m venv /work/envs/elm-ar
/work/envs/elm-ar/bin/pip install -r requirements-ar.txt
~~~

已有相同版本的环境可直接使用。以上是给复现者的安装命令，不会由仓库脚本自动执行。完整版本见 metadata/train_environment.json 和 metadata/ar_environment.json。

将另行提供的资产按清单中的相对路径放到一个目录，例如 /work/elm-assets。模型是**原始 Qwen3-4B**，不是 2507、3.5 或其他版本；逐文件 SHA 由资产清单核验。Oracle 阶段不需要拷贝原始 EEG 波形。

## 复现 80-token 基线

~~~bash
# 核验资产，创建全新输出目录；不启动 GPU，不修改资产目录。
# 先确保 /work/repro 已存在，并将 GPU 2,5 改成实际允许使用的物理卡号。
python3 -B tools/reproduce.py prepare \
  --assets /work/elm-assets \
  --output /work/repro/o80 \
  --train-env /work/envs/elm-train \
  --ar-env /work/envs/elm-ar \
  --gpus 2,5

# CPU 检查：初始化、题目/答案隔离、数据、模型与源码身份。
python3 -B tools/reproduce.py preflight --output /work/repro/o80

# 只有这一条命令会启动 GPU；不应在同一目录启动第二份。
python3 -B tools/reproduce.py run --output /work/repro/o80
~~~

入口生成独立工作区，沿用原训练、采样、损失和评测函数；仅适配路径、GPU 白名单，并跳过等待原机器已结束的 teacher 进程。不要直接执行 reference/ 中的历史脚本。

流程为 CPU 检查 → 30 步训练短测 → 20 题生成短测 → 初始 Val → 6 × 600 步训练及阶段评测 → 收尾报告。短测权重丢弃，正式训练使用固定初始化和新优化器；1800、3600 步保留完整 Val 对照。

每次只占白名单中的一张卡，训练入场需实际空闲显存 ≥26 GiB，生成需 ≥24 GiB；不要求空卡。总预算 12 小时、每阶段 4 小时，保留 8 GiB 物理显存和 12 GiB 磁盘余量。中断后在原目录执行同一条 run 命令，沿用原时钟和已提交 checkpoint，不重新计算预算。建议使用较短输出路径，避免推理引擎的 Unix socket 路径过长。

## 查看结果

输出目录下的 experiments/mainline_vrl/runs/oracle4b80_simple_20261024_s1014/ 包含：

- final_report.md：可读报告。
- results_summary.json：训练及各题型指标。
- label32_stat8/curriculum/oracle/：best、last、逐轮 checkpoint 和训练记录。
- label32_stat8/curriculum/evaluation/：逐题生成和阶段汇总。
- audit/final_verification.json：可用产物及进程退出检查。

报告分别列格式、逐题正确性、完整报告全对率，以及“明文原本正确的同题，Oracle 仍正确”的比例和支持数。这个实验只做 Val 诊断，没有新增 Test；EEG 对齐不会因门槛通过而自动启动。

## 数据与固定设置

当前题库包含 4691 条正确 Train 题、1744 条完整 Val 题。来源共 16 个 EEG 数据集：运动想象、情绪、注意力、疾病、临床 EEG 和睡眠；清单见 metadata/data_sources.json。真实数据 30 类，Oracle 合成组合 31 类，额外的 schizophrenia 类仅为合成。

题目是统计档位读取、两记录比较、选较高、选较低，外加类别/统计读取和完整报告。同档表示离散档位相同，不代表连续值相等。27B 仅生成题面，不参与最终答题。沿用受试者隔离和未见题面 Val 切分。

固定设置：seed 1014，batch 8 / micro 1，LR 0.001，weight decay 0.0001；正确 teacher 答案 CE + 0.5 全词表 forward KL + 0.0001 初始化约束。学生轨迹概率为 0 / 0.25 / 0.75 / 0.75 / 0.75 / 0.75，错误学生回答不作 CE 标签。20% 基础读取/报告、80% 四种简单操作，25% 更早训练池回放。按固定 ValTF240 CE 选 best，并单独报告 last1800 / last3600。

**验证范围：**源码快照、规则、路径迁移和使用现有私有资产的 CPU 预检已验证；本次发布没有重新运行 GPU 训练，也未验证 H20 端到端复现。硬件和推理运行时可能造成数值差异。
