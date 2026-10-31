# vLLM 0.32 推理环境

状态：已通过 A100 上的真实生成、输入一致性与独立报告验收。新旧版本有少量预测差异，见下表。旧 0.9.2 环境和历史结果保留。

## 环境分开

可微训练仍使用 Python 3.10、PyTorch 2.5.1/cu124 和原 Transformers 环境。
新版仅用于独立自回归生成与评测，使用 Python 3.11 和单独环境。
不要在旧环境内执行升级，也不要把 `requirements-train.txt` 装进新版推理环境。

本次采用 **vLLM 0.32.0+cu129、PyTorch 2.13.0+cu129、Transformers 5.19.0**。
[官方 0.32 发布页](https://github.com/vllm-project/vllm/releases/tag/v0.32.0)
列出的 PyPI 默认二进制是 CUDA 13；因此这里显式选择同一发布中的 CUDA 12.9
二进制及其完整 SHA，而不是把默认 vLLM wheel 和 cu129 Torch 混装。
CUDA、Torch、Transformers 是一组依赖；不要仅改 vLLM 的版本号。
文字/soft-token 入口还会间接导入 TorchCodec；这里固定官方 CPU 版 0.17.0，
避免其 PyPI 默认 CUDA 13 媒体库与 cu129 环境不匹配。我们不使用该包做 EEG 处理。

在授权 Linux 训练服务器执行，路径改为获准使用的私有目录。先创建临时目录，
避免下载器在目录不存在时退回系统临时目录。无需修改系统 CUDA 或驱动。

```bash
mkdir -p /work/elm-private/install-tmp /work/elm-private/pip-cache
export TMPDIR=/work/elm-private/install-tmp
export PIP_CACHE_DIR=/work/elm-private/pip-cache
export PIP_CONFIG_FILE=/dev/null

python3.11 -m venv /work/envs/elm-ar-v1
/work/envs/elm-ar-v1/bin/python -m pip install --no-cache-dir \
  --index-url https://mirrors.aliyun.com/pypi/simple \
  --extra-index-url https://download.pytorch.org/whl/cu129 \
  -r requirements-ar-v1.txt
/work/envs/elm-ar-v1/bin/python -m pip check
```

普通依赖使用阿里云镜像，Torch CUDA 构建使用官方 PyTorch 源，匹配的 vLLM wheel
使用官方发布文件。其 SHA 已固定在依赖文件中。目标机先核对 GPU/驱动；H20 尚未验收，
不要让 Agent 自动改驱动或借用其他人的服务。

## 显式选择后端

在新配置中设置：

```json
{
  "ar_backend": "v1",
  "ar_python": "/work/envs/elm-ar-v1/bin/python"
}
```

新后端移除旧的 `VLLM_USE_V1=0`。当前对照设置仍为 BF16、80-token soft prefix、
4096 上下文、最多 384 个新 tokens、温度 0、batch 16、eager 执行；显式关闭前缀缓存
和分块 prefill，先维持输入路径可比。不要同时更改模型、tokens、问题或评分。

`ar_backend: "legacy_v0"` 对应原 0.9.2。未写该键的旧配置继续走 legacy，
避免静默改变历史运行；历史 Oracle 复现器仍只用 `requirements-ar.txt` 的旧环境。
新旧环境在版本不符时会拒绝生成。

准备好模型、冻结 Oracle 和特征后，先做无 GPU 导入/API/文字输入检查：

```bash
CUDA_VISIBLE_DEVICES='' /work/envs/elm-ar-v1/bin/python -B \
  tools/check_inference_runtime.py --config /work/elm-private/config.json \
  --output /work/elm-private/CPU_inference_preflight.json
```

这一步不代表 GPU 生成成功。之后仍按运行指南执行真实 `evaluate` 和独立 `report`，
检查完整覆盖、原 parser、输入身份和进程退出。可选 `migration_text_controls: true`
会对 Val 中前两个不同类别做 native-ID/相同 word-embedding 生成对照；该诊断要求
Val 至少包含两个类别，不按答题准确率挑选样本。

## 验收边界

本次使用固定 6600 步 Projector、10200 步 Oracle、原始 Qwen3-4B 和同一批 32 条 Val EEG。
每个后端生成 256 个问题 × EEG/Oracle 两种输入，共 512 个回答，另做固定文字输入对照。
不新增优化步，不重新训练历史运行，不生成 Test。

升级会连带更换推理环境中的 Torch/Transformers；即使温度为零，也不能承诺跨环境
逐字完全一致。按同题记录事实正确性、格式、完整报告和输入/预测一致率。
共享 GPU 上的生成计时只是本次测量，不能当成纯 vLLM 版本的性能因果。
`peak_allocated_GiB` 仅记录调用进程的 Torch allocator；V1 的 EngineCore 在子进程，
因此该字段不包含完整引擎显存，不能用于新旧版本总显存比较。

## 本次结果

每项使用相同 32 条 Val EEG；统计单问是六项合计 192 问。这里只是固定小子集，
不是原 128 条完整 Val，也不是 Test。聚合记录见
[验收元数据](../metadata/vllm032_acceptance.json)，不包含逐人数据或生成原文。

| 输入 | 指标 | 0.9.2 V0 | 0.32 V1 |
|---|---|---:|---:|
| EEG Projector | 类别正确 | 12/32 | 12/32 |
| EEG Projector | 六项统计单问正确 | 125/192 | 127/192 |
| EEG Projector | 完整报告七项全对 | 1/32 | 1/32 |
| Oracle | 类别正确 | 32/32 | 32/32 |
| Oracle | 六项统计单问正确 | 190/192 | 190/192 |
| Oracle | 完整报告七项全对 | 27/32 | 25/32 |

- 两版各生成 512 个回答，另各生成 4 个文字输入控制回答，共 1032 个；全部独立复核评分，完整覆盖。各版两个固定 native-ID/embedding 控制的输出 token IDs 一致。
- 全部 512 个同题输入的 embedding SHA 一致；预测相同 504/512（98.4375%）。EEG 路径 6 个预测变化，Oracle 路径 2 个报告变化。Oracle 报告下降保留在表中，不将本次升级写成准确率提升。
- 两版唯一相同的格式错误均在 Oracle 统计单问：191/192 格式有效；EEG 路径及两版报告格式均有效。这一格式错误和事实错误分开记录。
- 本次实际生成调用耗时 132.50 → 116.34 秒，输出吞吐 58.15 → 66.25 tokens/s；引擎冷启动 29.15 → 53.69 秒。共享卡单次测量，不代表所有负载或 H20 的速度。
- 模型、数据、问题、评分和权重不变，没有新增优化步；所有相关 worker/EngineCore 已退出，角色与 GPU 锁已释放。训练的 35 步软件短测仍来自此前原训练环境的验收，本轮仅验新版推理。

新批次 Teacher 生成和 Val 评测须固定同一后端/版本；不同 loss 消融复用同一份冻结数据，
不要在组间切换生成版本。历史严格复现继续使用原环境。新版本的工程通过不等于
科学准确率提升，也不代表全量数据或 H20 已通过。
