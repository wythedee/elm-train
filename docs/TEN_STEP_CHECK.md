# 2026-11-02 十步流程检查

状态：JS5真实完成，`passed_bounded_10step_pipeline`；随后独立CPU复算通过。
没有在Mac做模型优化，没有重新开启历史run。

| 阶段 | 实际完成 |
|---|---|
| 同一冻结4B文字Teacher | 120Train＋40Val，共160条，10个生成batch；95条Train答对后用于监督，全部40Val保留 |
| fresh Oracle | 10次optimizer更新，batch8；CE＋0.5全文词表forward KL＋0.0001初始值锚定；Decoder冻结 |
| Oracle自回归生成 | 40条声明Val全部生成并独立解析；包含基础读取、报告和四种简单操作 |
| fresh EEG Projector | 10次optimizer更新，batch8；使用本轮Oracle10冻结目标；SFT＋token对齐，Decoder/Encoder输入特征冻结 |
| EEG自回归生成与报告 | 80题×Projector/Oracle两种输入，共160回答，生成后CPU独立报告 |

两段训练共160次样本访问均按固定seed独立重算一致。Oracle优化器2个参数状态、
Projector优化器20个参数状态均为step10；模型保存/重载、全部随机状态重载、
全部80个位置梯度检查通过。没有OOM或NaN；Projector峰值Torch显存约13.24GiB。
原始回答按程序金标/原parser和summary复算通过，GPU worker/EngineCore退出，role/GPU锁free。

输入为16来源各Train/Val两条，共32/32条已有冻结Encoder特征；选取与重分片保留原张量。
本轮没有重新跑原始EEG预处理或Encoder forward，这部分此前已有独立A100工程验收。
题面复用既有Qwen3.8-27B的96种生成结果，未重新出题或按本轮分数更换题面。

本轮使用JS5既有训练环境和vLLM0.9.2，不能称H20或新vLLM0.32 Teacher入口已验证。
新0.32的独立推理另见[VLLM_UPGRADE.md](VLLM_UPGRADE.md)。这是软件检查，
不代表10步模型效果好、完整Val/Test完成或正式长训稳定；Test=N/A，未测试on-policy分支。

运行命令在[README](../README.md)。
服务器产物：`/data/yuqiudeng/vstats-elm/outputs/pipeline10_20261102_v1/`，
含`final_verification.json`、`independent_verification.json`、`final_report.md`和各阶段日志。
逐人数据、raw回答和checkpoint保留私有服务器；本仓库只公开代码、通用题面和聚合验算结论。
