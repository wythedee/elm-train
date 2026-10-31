# 师姐端 / H20 运行交接单

**第一项任务：在一张获准使用的 H20 上跑通软件短测，再评测已提供的6600步模型，交回报告。**
当前仓库的可执行入口是 `elm_pipeline.cli`；大批次Oracle训练还不能直接启动。
不要求师姐端重新设计任务、猜训练数据格式或修补缺失的大批次训练器。

## 双方分工

| 谁 | 负责什么 |
|---|---|
| 我们这边维护代码 | 固定模型/标签/统计语义，提供权重、可执行代码、题库构建与Teacher筛选、实验配置及报告口径；修复交回的实现问题 |
| 师姐端的运行Agent | 清点本地数据和已分配H20，配置路径，执行下面已实现的步骤，记录实际产物和失败，交回报告；之后执行已就绪的实验清单 |
| 数据/算力负责人 | 提供数据版本、可用路径、GPU白名单和长任务额度；已明确的设置无需重复确认 |

## 开跑前必须拿到的信息

把下面信息写进私有工作区的运行记录，不能根据我们的旧绝对路径猜测：

- GPU型号、驱动、实际空闲显存、允许用的物理卡号；模型任务仅在Linux服务器执行。
- 私有工作目录，及训练Python3.10 / 推理Python3.11环境路径。安装与CUDA配套见 [运行指南](AGENT_RUNBOOK.md#environment)。
- 原始数据或Native200的位置、版本和标签；先选择至少一个版本已核对的来源，优先ADFTD。
- 若只有原始数据，按 [数据表](DATASETS.md) 核对相应adapter。九个入口仅ADFTD完成过本地raw验收，另外七个raw路线尚未全部封装；缺少入口时记录缺口，不换一个同名数据顶替。

有兼容的16来源时可各取Train/Val两条做工程检查；只有ADFTD时先做单来源，报告如实写出范围。
这不是全量数据训练，也不是正式泛化结论。遇到同人跨split等身份问题，不静默修改旧清单。

## 执行顺序

1. **记录代码和静态检查。** clone最新仓库，保存commit；运行README的四条静态检查。不要直接执行`reference/`历史脚本。
2. **下载并核验模型。** 按 [权重指南](CHECKPOINTS.md) 下载四份组件和配套文件，再单独下载原始Qwen3-4B。公开下载不需要我们提供的访问令牌。
3. **准备数据。** 使用兼容Native200，或按数据指南先做所需raw小测；记录窗口/人/类别数、坏记录、通道和单位。不要用旧特征缓存冒充raw入口验收。
4. **准备两个私有配置。** 按下表复制模板并修改真实路径；两个任务使用相同数据选择和冻结资产，输出目录分别新建，`gpus`都写获准的单张卡。
5. **按下面命令跑fresh软件短测。** 新Projector训练0→30，退出后在同一配置严格恢复到35；实际生成和独立报告。它只验证实现，不以35步准确率判定模型优劣。
6. **单独评测6600模型。** 复用步骤5已经完成的同版本特征；使用另一个输出目录，加载发布的6600权重，只生成和报告，不再训练。
7. **交回报告，先结束本次短测。** 不把`NEXT_DAY_PLAN`中的建议数字当作已经可运行的长任务清单。

| 配置 | 需要修改 / 保持 |
|---|---|
| `configs/pipeline_debug.example.json` → 私有`smoke35.json` | 新输出和特征目录、真实manifest/data/path_map、数据集和GPU、两个Python路径；保留35步/seed1014/80tokens/原Loss/发布schema与阈值；不设置initial_projector或evaluation_checkpoint |
| `configs/pipeline_report6600.example.json` → 私有`report6600.json` | 新输出目录；features指向上述已完成特征；其余数据和资产与短测一致；保留train_steps=0和发布的evaluation_checkpoint |

短测模板上限为总2小时、每阶段40分钟，含排队；中断不重置时钟。这里没有新增研究训练预算。
开始前记下配置，过程中不要改预算/源码/数据版本来冒充同一次严格恢复。

下面的路径仅为例子，替换成实际私有工作区；环境按运行指南准备。

```bash
ELM_WORK=/work/elm-private/h20_acceptance
ELM_TRAIN_PY=/work/envs/elm-train/bin/python
ELM_AR_PY=/work/envs/elm-ar-v1/bin/python

# smoke35.json中先填好新目录和真实路径。以下命令从Git checkout运行。
"$ELM_TRAIN_PY" -B -m elm_pipeline.cli --config "$ELM_WORK/smoke35.json" check
"$ELM_TRAIN_PY" -B -m elm_pipeline.cli --config "$ELM_WORK/smoke35.json" features
CUDA_VISIBLE_DEVICES='' "$ELM_AR_PY" -B tools/check_inference_runtime.py \
  --config "$ELM_WORK/smoke35.json" --output "$ELM_WORK/CPU_inference_preflight.json"
"$ELM_TRAIN_PY" -B -m elm_pipeline.cli --config "$ELM_WORK/smoke35.json" train --stop-at 30
"$ELM_TRAIN_PY" -B -m elm_pipeline.cli --config "$ELM_WORK/smoke35.json" train --stop-at 35
"$ELM_TRAIN_PY" -B -m elm_pipeline.cli --config "$ELM_WORK/smoke35.json" evaluate --tag smoke35
CUDA_VISIBLE_DEVICES='' "$ELM_TRAIN_PY" -B -m elm_pipeline.cli \
  --config "$ELM_WORK/smoke35.json" report

# report6600.json指向同一已完成特征，另用全新输出目录；不调用train或重复features。
"$ELM_TRAIN_PY" -B -m elm_pipeline.cli --config "$ELM_WORK/report6600.json" check
"$ELM_TRAIN_PY" -B -m elm_pipeline.cli --config "$ELM_WORK/report6600.json" evaluate --tag report6600
CUDA_VISIBLE_DEVICES='' "$ELM_TRAIN_PY" -B -m elm_pipeline.cli \
  --config "$ELM_WORK/report6600.json" report
```

## 怎样算工程通过

- 配置/模型身份检查通过；选中的数据全部能定位，特征shape、mask、标签/统计顺序正确，坏记录和覆盖有记录。
- `train_result_step30.json`和`train_result_step35.json`存在；后者`start_step=30`、提交步35，全部参数/80槽收到梯度，Decoder和冻结特征输入保持冻结，优化器/RNG保存恢复检查通过。
- 两次评测都覆盖所选Val的所有声明问题及Oracle正控，原始输出能够独立评分；报告分开列类别、六统计、七项报告全对、格式和支持数。
- `final_verification.json`中的`process_exit_check_pending=false`、原始parser复核完成；相关worker/EngineCore退出，角色/GPU锁释放。不能把未来全量阶段的缺失改成“已完成”。

**不额外设置准确率启动门槛。** 输出格式正常也不等于事实正确，低准确率和失败同样交回。
35步fresh模型与6600模型的比较仅用于检查两条加载路径；不要称为同预算科学对照。
不同GPU/推理栈会产生数值差异，禁止改题、筛掉难Val或重写raw来追求与旧报告逐字一致。

## 遇到这些情况就停止相应阶段并报告

- 原始版本/标签/split不明、缺少adapter或真实数据；已完成的下载/静态检查仍可交回，不造样本凑条数。
- 环境/驱动不兼容、OOM、NaN、缺梯度、模型身份变化、恢复状态不一致或时间/磁盘保护触发。保留日志和partial，不自动换模型、Loss或升级系统驱动。
- 旧worker/EngineCore或锁未释放时不重复派发；不终止其他人的进程，不重新开启已关闭运行，不延长原时钟。

报告使用`passed_bounded_H20_software`、`blocked_data`、`blocked_environment`或`failed_execution`等清楚的状态，并写实际完成到哪一步。

## 交回什么

在私有交付目录整理一份`RUN_REPORT.md`，首屏写：

1. 状态、代码commit、H20/驱动/两套包版本、单卡运行时间和失败原因。
2. 实际数据来源/版本、Train/Val窗口和人数、标签覆盖、预处理/特征是否新生成；与我们的选择是否相同。
3. fresh35与report6600各自的类别、六统计、报告七项全对、格式、支持数；Oracle正控另列。数据选择不同就不直接比较旧128条Val的分数。
4. 冻结/梯度/严格恢复检查和吞吐/显存信息；仍缺的来源、阶段或输入。

把以下证据与报告一起私下交回用户，不提交到公开GitHub或ModelScope：

- 两份实际配置、代码commit和包版本；输入身份audit/数量、`features_result.json`、`accepted_coverage.json`。
- 两次训练的`train_result_step*.json`、`commit.json`、`train_trace.jsonl`及阶段日志。
- 两个输出目录的`final_report.md`、`results_summary.json`、`final_verification.json`；各评测目录的`identity.json`、`summary.json`、`raw.jsonl`及错误/partial记录。

首轮报告包不必包含EEG、环境、缓存、下载的权重或大checkpoint；新checkpoint先保留在H20工作区供后续定位。
不要把逐人预测或凭据发到公开仓库。

## 之后怎样批量跑

长任务就绪后，会单独提供固定题库/Teacher筛选记录、case配置、seed、共同评测和预算。
师姐端按那份清单使用尽可能多的已分配GPU，一任务一卡、独立配置/输出/缓存，完成一组补下一组；
同一任务的训练/恢复/评测顺序执行。不要把候选GPU列表当成DDP，不用`torchrun`复制同一个任务。

当前10万候选题、3万步、三组×两seed仍是提案：新数据生成/筛选、变体训练和批量启动器未全部接齐，
这些由我们维护方补齐。历史`tools/reproduce.py`是固定3600步复现且缺另外的题库资产，不能用它冒充新批次。
师姐端不负责通过修改旧脚本、重用已训练Oracle作fresh初始化、放宽覆盖或自行追加预算来填这个缺口。
