#!/usr/bin/env python3
"""New simple Oracle task IO; question wording comes from verified27B artifacts."""

if __name__ == "__main__":
    raise SystemExit("Reference snapshot only. Use tools/workbench.py for inspection; configure a fresh run before training.")

import copy,hashlib,importlib.util,json,re,sys
from pathlib import Path
PLAN=Path(__file__).resolve().parent
P=PLAN.parents[4]
S=P/"experiments/mainline_vrl/scripts/python"
if str(S) not in sys.path:sys.path.insert(0,str(S))
import oracle_curriculum28_tasks_1016_v3 as legacy
FIELDS=legacy.FIELDS
LEVELS=["low","mid","high"]
SYSTEM=("Read the supplied EEG statistical facts. Each value is an ordinal bin, ordered low < mid < high. "
"Compare bins, not continuous values. 'equal' or 'tie' means the same bin. "
"Answer the question using exactly one JSON object with one key, answer. Do not add an explanation.")
def dumps(v):return json.dumps(v,ensure_ascii=False,separators=(",",":"))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
_CACHED_VALIDATOR=None
def validator():
 global _CACHED_VALIDATOR
 if _CACHED_VALIDATOR is not None:return _CACHED_VALIDATOR
 s=importlib.util.spec_from_file_location("simple_task_gold",PLAN/"simple_task_validator_v1.py")
 m=importlib.util.module_from_spec(s);s.loader.exec_module(m);_CACHED_VALIDATOR=m;return m
def expected(e):
 if e["kind"]=="base":return copy.deepcopy(e["q"]["answer"])
 assert e["kind"]=="simple"
 return validator().gold(e["operation"],[r["bins"][e["field"]] for r in e["records"]])
def target(e):return dumps(expected(e))
def question(e):
 return e["q"]["question"] if e["kind"]=="base" else e["template_wording"].format(field=e["field"],record_a="A",record_b="B")
def messages(e,teacher=False):
 assert len(e["records"]) in [1,2]
 pieces=[]
 for j,r in enumerate(e["records"]):
  # Teacher sees observed facts only; student gets tokens selected by its own label and each field bin.
  fact=dict(label=r["label"],statistics={f:r["bins"][f] for f in FIELDS})
  pieces.append("Record "+chr(65+j)+":\n"+(dumps(fact) if teacher else "__SIMPLE_ORACLE_"+str(j)+"__"))
 body="EEG statistical facts:\n"+"\n".join(pieces)+"\nQuestion: "+question(e)
 if e["kind"]=="simple":
  body+="\nAllowed answer strings: "+dumps(validator().OPS[e["operation"]]["result_values"])+'\nOutput schema: {"answer":"<one allowed string>"}'
 system=SYSTEM if e["kind"]=="simple" else "Read the supplied EEG observed facts. Return only the requested JSON, without extra fields or diagnoses."
 assert "__SIMPLE_ORACLE_" not in question(e)
 return [dict(role="system",content=system),dict(role="user",content=body)]
def render(tok,emb,e,prefix=None,teacher=False,reference=False):
 assert reference is False,"Reference answers must never enter the teacher or student prompt"
 import torch
 text=tok.apply_chat_template(messages(e,teacher),tokenize=False,add_generation_prompt=True,enable_thinking=False)
 if teacher:
  assert prefix is None
  ids=tok(text,add_special_tokens=False)["input_ids"]
  return emb(torch.tensor(ids,device=emb.weight.device,dtype=torch.long)).detach()
 assert prefix is not None and len(prefix)==len(e["records"])
 parts=[]
 for p in re.split(r"(__SIMPLE_ORACLE_\d+__)",text):
  m=re.fullmatch(r"__SIMPLE_ORACLE_(\d+)__",p)
  if m:parts.append(prefix[int(m.group(1))].to(device=emb.weight.device,dtype=emb.weight.dtype))
  elif p:parts.append(emb(torch.tensor(tok(p,add_special_tokens=False)["input_ids"],device=emb.weight.device,dtype=torch.long)).detach())
 assert len(parts)>1
 result=torch.cat(parts)
 assert not any("__SIMPLE_ORACLE_" in p for p in re.split(r"(__SIMPLE_ORACLE_\d+__)",text) if p and not re.fullmatch(r"__SIMPLE_ORACLE_(\d+)__",p))
 return result
def parse(text,e):
 if e["kind"]=="base":return legacy.parse(text,e)
 v=validator();levels=[r["bins"][e["field"]] for r in e["records"]]
 score=v.score(text,e["operation"],levels)
 try:prediction=json.loads(text)
 except (ValueError,TypeError):prediction=None
 return dict(prediction=prediction,valid=score["valid"],correct=score["correct"],
  fact_score=None,decision_score=float(score["correct"]) if e["operation"]!="read_level" else None,
  trace_correct=None,fact_items=[],single_answer_no_explicit_fact_trace=True)
def fixture():
 v=validator();counts=0
 import itertools
 for op,meta in v.OPS.items():
  for levels in itertools.product(LEVELS,repeat=meta["records"]):
   # Deliberately non-natural fixture string is never included in training or evaluation.
   e=dict(kind="simple",operation=op,field=FIELDS[0],template_wording="{record_a} {field} {record_b} fixture" if meta["records"]==2 else "{record_a} {field} fixture",
    records=[dict(id="FIXTURE_"+str(i),label="FIXTURE_LABEL",bins={f:level for f in FIELDS}) for i,level in enumerate(levels)])
   assert parse(target(e),e)["correct"]
   inp=messages(e);wrong=copy.deepcopy(e);wrong["expected"]={"wrong":True};wrong["teacher_trajectory"]="INVALID_REFERENCE"
   assert messages(wrong)==inp and messages(wrong,True)==messages(e,True)
   changed=copy.deepcopy(e)
   for r in changed["records"]:r["label"]="CHANGED";r["bins"]={f:"high" for f in FIELDS}
   assert messages(changed)==inp
   assert not any(r["label"] in inp[1]["content"] for r in e["records"])
   counts+=1
 return dict(status="cpu_fixture_passed",truth_cases=counts,changed_expected_or_teacher_does_not_change_input=True,
  changed_labels_or_bins_do_not_change_student_input=True,model_loaded=False,generation_started=False,
  fresh_teacher_pool_not_yet_built=True,source_sha256=sha(__file__))
if __name__=="__main__":
 print(dumps(fixture()))
