# EEG Oracle 4B

将类别和六项统计档位编码为可训练soft tokens，让冻结Qwen3-4B回答问题；Oracle验证达标后，再训练EEG侧投影器逐token对齐。

本仓库保存已执行的80-token/3600步代码及依赖、任务规则和下一批实验规格。原始代码的函数与类AST逐项不变；增加入口保护，避免直接运行历史脚本写回已关闭的实验目录。

## 使用

~~~bash
python3 tools/workbench.py check
python3 tools/workbench.py plan
python3 tools/workbench.py data-info
python3 tools/workbench.py assets --workspace /path/to/vstats-elm
~~~

check仅用Python标准库验证源文件哈希、语法和30种统计档位真值/交换方向/同档/JSON规则，不需要GPU或数据。
assets只检查已有工作区的数据/模型文件是否齐全；加--hash做完整SHA校验。

## 代码范围

- reference/experiments/mainline_vrl/scripts/python/oracle4b80_simple_1024_v1.py：已执行的Oracle训练与自回归评测。
- reference内simple_oracle_tasks_v2.py、simple_task_validator_v1.py：题目输入、程序金标和解析规则。
- reference内eeg_projector80_4b_v1.py、eeg_align4b80_simple_1024_v1.py：条件EEG对齐准备代码；只完成CPU检查，尚无GPU训练结果。
- metadata/source_manifest.json：每个源文件的原始/归档哈希和变换说明。
- metadata/*environment.json：实际环境版本记录，不包含环境文件。
- configs/next_batch.json：原配方、报告权重x4、160tokens、只做答案监督的四组规格。

**这是可审查的代码快照，不是已经验证的一键部署包。**新消融的通用训练启动器尚未实现；四组配置不会自动启动训练。实际运行需要另建run、配置数据路径并先通过对应GPU Smoke。默认命令只做检查，不重开历史run。

## 数据

Oracle阶段使用从16个EEG数据集提取的类别和六项统计档位清单，以及27B题面和4B无参考正确训练答案。该阶段不直接读取EEG波形。
实际使用4691条正确Train题、1744条完整Val题；真实30类，额外一类仅合成。分组仍按原受试者和题面分开。
后续EEG对齐才需要已有EEG专家特征；原始波形、受试者逐条清单、题库输出、teacher答案、模型、初始化和checkpoint均由外部工作区提供。

数据集列表与用途见metadata/data_sources.json，私有资产路径和身份见metadata/asset_requirements.json。本Git仓库只包含代码、规则、规格及不含逐条数据的元信息。

## 评测

保留完整Val，并单列“明文原本正确的同题，Oracle仍正确”的支持数和正确率；报告整份全对、格式错误、逐字段错误。此诊断未生成新Test。
不同token数同时改变编码容量和位置，不能声称纯长度因果。新消融统一报告3600步结果，并单独列预定ValTF选模结果，不能混报。
