#!/usr/bin/env python3
"""Inspect this code-only release. Never launches training or writes an existing run."""
from pathlib import Path
import argparse,ast,hashlib,importlib.util,itertools,json,re,sys
ROOT=Path(__file__).resolve().parents[1]
def read(path):return json.loads(Path(path).read_text())
def digest(path):
 h=hashlib.sha256()
 with Path(path).open("rb") as f:
  for chunk in iter(lambda:f.read(1024*1024),b""):h.update(chunk)
 return h.hexdigest()
def check():
 manifest=read(ROOT/"metadata/source_manifest.json")
 for row in manifest["source_files"]:
  p=ROOT/row["snapshot_path"];assert p.is_file() and digest(p)==row["snapshot_sha256"],str(p)
  if p.suffix==".py":ast.parse(p.read_text())
 current=ROOT/"metadata/current_pipeline_provenance.json"
 if current.exists():
  for rel,want in read(current)["source_sha256"].items():
   p=(ROOT/rel).resolve();assert ROOT in p.parents and digest(p)==want,"Current pipeline source mismatch: "+rel
 badsuffix={".pt",".pth",".ckpt",".safetensors",".h5",".hdf5",".lmdb",".npy",".npz",".jsonl"}
 secret=re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----|gh[pousr]_[A-Za-z0-9]{24,}|hf_[A-Za-z0-9]{30,}|AKIA[0-9A-Z]{16}")
 paths=[p for p in ROOT.rglob("*") if p.is_file() and ".git" not in p.relative_to(ROOT).parts and "__pycache__" not in p.relative_to(ROOT).parts]
 for p in paths:
  assert p.suffix not in badsuffix and p.stat().st_size<1024*1024,str(p)
  assert not secret.search(p.read_text()),"Potential credential in "+str(p)
  if p.suffix==".py":compile(p.read_text(),str(p),"exec",dont_inherit=True)
 candidates=list((ROOT/"reference").rglob("simple_task_validator_v1.py"));assert len(candidates)==1
 sp=importlib.util.spec_from_file_location("released_ordinal_rules",candidates[0]);v=importlib.util.module_from_spec(sp);sp.loader.exec_module(v)
 tests=0;ranks={"low":0,"mid":1,"high":2}
 for op in ["read_level","compare_level","choose_higher","choose_lower"]:
  count=1 if op=="read_level" else 2
  for levels in itertools.product(["low","mid","high"],repeat=count):
   if op=="read_level":answer=levels[0]
   elif ranks[levels[0]]==ranks[levels[1]]:answer="equal" if op=="compare_level" else "tie"
   elif op=="compare_level":answer="lower" if ranks[levels[0]]<ranks[levels[1]] else "higher"
   else:
    chooseA=(ranks[levels[0]]>ranks[levels[1]]) if op=="choose_higher" else (ranks[levels[0]]<ranks[levels[1]])
    answer="A" if chooseA else "B"
   assert v.gold(op,list(levels))=={"answer":answer}
   assert v.score(json.dumps({"answer":answer}),op,list(levels))["correct"]
   assert not v.score(json.dumps({"answer":answer,"extra":0}),op,list(levels))["valid"]
   tests+=1
 cfg=read(ROOT/"configs/next_batch.json")
 for case in cfg["cases"].values():assert case["total_tokens"]==case["label_tokens"]+6*case["stat_tokens_each"]
 assert cfg["common"]["steps"]==3600 and cfg["common"]["seed"]==1014
 print(json.dumps(dict(status="passed",source_files=len(manifest["source_files"]),code_package_files=len(paths),
  ordinal_truth_cases=tests,no_private_data_or_weights=True,credential_scan_passed=True,
  all_source_SHA_and_AST_syntax_passed=True,GPU_used=False,training_started=False)))
def assets(workspace,full):
 want=read(ROOT/"metadata/asset_requirements.json")["files"];missing=[];changed=[];total=0
 workspace=Path(workspace).resolve()
 for rel,meta in want.items():
  p=workspace/rel
  if not p.is_file():missing.append(rel);continue
  total+=p.stat().st_size
  if full and digest(p)!=meta["sha256"]:changed.append(rel)
 print(json.dumps(dict(workspace=str(workspace),expected_files=len(want),missing=missing,changed=changed,
  bytes_of_available_files=total,full_SHA_checked=full,no_files_written=True)))
 return 1 if missing or changed else 0
def main():
 a=argparse.ArgumentParser();a.add_argument("command",choices=["check","plan","data-info","assets"])
 a.add_argument("--workspace");a.add_argument("--hash",action="store_true");args=a.parse_args()
 if args.command=="check":check()
 elif args.command=="plan":print(json.dumps(read(ROOT/"configs/next_batch.json"),ensure_ascii=False,indent=2))
 elif args.command=="data-info":print(json.dumps(read(ROOT/"metadata/data_sources.json"),ensure_ascii=False,indent=2))
 else:
  if not args.workspace:a.error("--workspace is required for assets")
  return assets(args.workspace,args.hash)
 return 0
if __name__=="__main__":raise SystemExit(main())
