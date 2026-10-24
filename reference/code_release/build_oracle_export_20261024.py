
if __name__ == "__main__":
    raise SystemExit("Reference snapshot only. Use tools/workbench.py for inspection; configure a fresh run before training.")

from pathlib import Path
import ast,hashlib,importlib.metadata as md,json,shutil,subprocess,sys,time
P=Path("/data/yuqiudeng/vstats-elm")
sys.path.insert(0,str(P/"experiments/mainline_vrl/scripts/python"))
import oracle4b80_simple_1024_v1 as m
D=P/"experiments/mainline_vrl/plans/diagnostics/oracle4b_learnability_20261024_s1014"
U=m.ROOT;R=P/"code_release/eeg-oracle-workbench"
assert not R.exists(),"Never overwrite an existing code export"
R.mkdir(parents=True);sources=set()
for mod in list(sys.modules.values()):
 f=getattr(mod,"__file__",None)
 if not f:continue
 f=Path(f).resolve()
 if f.is_file() and P in f.parents and f.suffix==".py" and not any(x in f.parts for x in [".venv","runtime"]):sources.add(f)
for rel in m.config()["dependencies"]:sources.add(P/rel)
sources.update([Path(m.__file__),P/"experiments/mainline_vrl/scripts/python/eeg_align4b80_simple_1024_v1.py",D/"eeg_projector80_4b_v1.py"])
def sha(f):return hashlib.sha256(Path(f).read_bytes()).hexdigest()
manifest=[]
for f in sorted(sources):
 rel=f.relative_to(P);out=R/"reference"/rel;out.parent.mkdir(parents=True,exist_ok=True);original=f.read_text()
 if f.suffix==".py":
  tree=ast.parse(original);anchor=1 if original.startswith("#!") else 0
  for node in tree.body:
   if isinstance(node,ast.Expr) and isinstance(node.value,ast.Constant) and isinstance(node.value.value,str) and node is tree.body[0]:anchor=max(anchor,node.end_lineno);continue
   if isinstance(node,ast.ImportFrom) and node.module=="__future__":anchor=max(anchor,node.end_lineno);continue
   break
  lines=original.splitlines(keepends=True)
  lines.insert(anchor,'\nif __name__ == "__main__":\n    raise SystemExit("Reference snapshot only. Use tools/workbench.py for inspection; configure a fresh run before training.")\n\n')
  out.write_text("".join(lines));archived=ast.parse(out.read_text())
  def definitions(t):return {x.name:ast.dump(x,include_attributes=False) for x in t.body if isinstance(x,(ast.FunctionDef,ast.AsyncFunctionDef,ast.ClassDef))}
  assert definitions(tree)==definitions(archived)
 else:shutil.copy2(f,out)
 manifest.append(dict(upstream_path=str(rel),original_sha256=sha(f),snapshot_path=str(out.relative_to(R)),snapshot_sha256=sha(out),all_function_and_class_AST_unchanged=True,entrypoint_guard_added=f.suffix==".py"))
for folder in ["metadata","configs","tools"]:(R/folder).mkdir()
(R/"metadata/source_manifest.json").write_text(json.dumps(dict(schema_version=1,baseline_run=U.name,baseline_source_sha256=sha(Path(m.__file__)),baseline_protocol_sha256=sha(U/"protocol.json"),source_files=manifest,no_data_or_weights_included=True),indent=2)+"\n")
proto=m.config();requirements={}
for role,items in [("model",proto["models"]),("initialization",proto["assets"])]:
 for path,h in items.items():
  f=Path(path)
  if f.suffix==".py":continue
  requirements[str(f.relative_to(P))]=dict(role=role,sha256=h)
for path,h in m.read(U/"audit/dispatch_freeze.json")["teacher_and_plans"].items():requirements[str(Path(path).relative_to(P))]=dict(role="private_task_or_teacher_data",sha256=h)
for split in ["train","val"]:
 f=m.DATA/(split+"_records.json");requirements[str(f.relative_to(P))]=dict(role="private_EEG_derived_metadata",sha256=sha(f))
(R/"metadata/asset_requirements.json").write_text(json.dumps(dict(files=requirements,waveform_required_for_Oracle_stage=False,original_EEG_features_required_for_later_alignment=True),indent=2)+"\n")
info=dict(protocol="Native200, original16-domain subject-disjoint Train/Val; no newTest in this diagnostic",datasets=list(proto["data"]["original_ids"]["train"]["dataset_counts"]),
 record_bank_counts={s:proto["data"]["original_ids"][s]["samples"] for s in ["train","val"]},Oracle_training_candidates=5608,accepted_correct_unassisted4B_training_examples=4691,validation_queries=1744,real_classes=30,synthetic_Oracle_classes=31,synthetic_only_extra_class="schizophrenia",ordinal_fields=list(m.tasks.FIELDS),ordinal_levels=["low","mid","high"],operations=["read_level","compare_level","choose_higher","choose_lower"],same_bin_is_not_equal_continuous_value=True,current_stage="Oracle facts-to-soft-prefix diagnostic; no rawwaveform as Oracle forwardinput; EEG alignment not started")
(R/"metadata/data_sources.json").write_text(json.dumps(info,ensure_ascii=False,indent=2)+"\n")
versions={}
for package in ["torch","transformers","numpy","pandas","scipy","scikit-learn","einops","h5py","mne","timm","safetensors","tokenizers"]:
 try:versions[package]=md.version(package)
 except md.PackageNotFoundError:pass
(R/"metadata/train_environment.json").write_text(json.dumps(dict(python=sys.version.split()[0],packages=versions,no_packages_installed=True,cuda_visible_device_admission="actualfree26GiB train/24GiB AR,3samples10secondsapart;8GiB physicalreserve,12GiB diskreserve",copied_environment_not_included=True),indent=2)+"\n")
common=dict(model="original Qwen3-4B frozen",steps=3600,rounds=6,steps_per_round=600,seed=1014,batch_size=8,micro_batch_size=1,learning_rate=.001,weight_decay=.0001,anchor_weight=.0001,teacher="Same closedbaseline unassisted4B correctTrain examples; all1744Val queries retained",subject_split="Unchanged original16-domain subject and sampleIDs",native_initializer="Freshnative4B semantic init; no trainedweightwarmstart",primary_comparison_checkpoint="last3600",secondary_checkpoint="fixedValTF240CE selectedbest; separatelylabelled",gate="Fourtypes text-correct Valexact>=.90,support>=16;fullreport exact>=.90;frozencoverage")
cases={"A_baseline":dict(label_tokens=32,stat_tokens_each=8,total_tokens=80,KL_coefficient=.5,report_objective_multiplier=1,role="Closedbaseline reference on currentA100 server; no automatic rerun"),
"B_report4":dict(label_tokens=32,stat_tokens_each=8,total_tokens=80,KL_coefficient=.5,report_objective_multiplier=4,role="Proposed:report example(CE+.5KL)x4; unchanged unweightedanchor"),
"C_tokens160":dict(label_tokens=64,stat_tokens_each=16,total_tokens=160,KL_coefficient=.5,report_objective_multiplier=1,role="Proposed:fixedlabel/stat quota ratio doubled;capacity andpositionscoupled,notpurelengthcausality"),
"D_CEonly":dict(label_tokens=32,stat_tokens_each=8,total_tokens=80,KL_coefficient=0,report_objective_multiplier=1,role="Proposed:correctteacher answerCE+sameanchor;noKLorstudentrolloutCE")}
(R/"configs/next_batch.json").write_text(json.dumps(dict(status="proposal_not_launched",common=common,cases=cases,baseline_snapshot_verified=True,portable_training_launcher_implemented=False,new_variant_GPU_smoke_and_explicit_launch_still_required=True,EEG_not_auto_dispatched_from_this_code_bundle=True),ensure_ascii=False,indent=2)+"\n")
(R/".gitignore").write_text(".venv/\n__pycache__/\n*.py[cod]\n.env\n.env.*\n*.log\n*.pid\n*.lock\n*.pt\n*.pth\n*.ckpt\n*.safetensors\n*.h5\n*.hdf5\n*.lmdb\n*.npy\n*.npz\n*.jsonl\ndata/\nassets/\nmodels/\nruns/\ncache/\noutputs/\n")
(R/"README.md").write_text("""# EEG Oracle 4B

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
""")
head=subprocess.check_output(["git","rev-parse","HEAD"],cwd=P,text=True).strip()
(R/"metadata/provenance.json").write_text(json.dumps(dict(created_time=time.time(),upstream_git_HEAD=head,baseline_run=U.name,baseline_closed=True,source_snapshot_only=True,training_code_original_source_sha256=sha(Path(m.__file__)),no_private_data_weights_environment_logs_or_per_sample_outputs_exported=True,no_current_or_old_run_modified=True,upstream_dirty_changes_not_committed=True),indent=2)+"\n")
print(json.dumps(dict(export=str(R),files=len(manifest),source_bytes=sum((R/x["snapshot_path"]).stat().st_size for x in manifest),train_versions=versions)))
