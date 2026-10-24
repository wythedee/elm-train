#!/usr/bin/env python3
"""Frozen-model statistical reasoning and continuous-sleep probes. No training."""
from __future__ import annotations

if __name__ == "__main__":
    raise SystemExit("Reference snapshot only. Use tools/workbench.py for inspection; configure a fresh run before training.")

from pathlib import Path
import argparse,collections,copy,fcntl,hashlib,itertools,json,math,os,random,shutil,signal,statistics,subprocess,sys,time,traceback
P=Path(__file__).resolve().parents[4]
R=P/'experiments/mainline_vrl/runs/reason_sleep_20261015_s1014'
J=P/'experiments/mainline_vrl/runs/stat_joint28_20261015_s1014'
C=P/'experiments/mainline_vrl/runs/stat_compact28_20261015_s1014'
S=P/'experiments/mainline_vrl/runs/stat_sleep_20261014_s1014/sleep/epochs32_native_init'
D=P/'experiments/mainline_vrl/runs/stat_sleep_20261014_s1014/statistics/data'
SD=S.parent/'data'
ARPY='/home/workspace/yuqiudeng/envs/eeglm-vllm/bin/python'
TRAINPY=str(P/'.venv/bin/python')
MODEL=P/'experiments/eeg_vrl_demo/models/Qwen/Qwen3-1___7B'
FIELDS=['alpha_relative_power','theta_beta_ratio','spectral_entropy','hjorth_mobility','hjorth_complexity','line_length']
VALUES=['low','mid','high'];STAGES=['wake','n1','n2','n3','rem','unknown'];SEED=1014
mean=statistics.mean
def read(p):return json.loads(Path(p).read_text())
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
def write(p,x):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n');tmp.replace(p)
def guard():
 if shutil.disk_usage(R).free<12*2**30:raise RuntimeError('Disk guard: minimum12GiB remains')
def rngkey(x):return random.Random(int(hashlib.sha256((str(SEED)+str(x)).encode()).hexdigest()[:12],16))
def relation(a,b):return 'lower' if VALUES.index(a)<VALUES.index(b) else 'higher' if VALUES.index(a)>VALUES.index(b) else 'equal'
def derive(levels,q):
 family=q['family'];desired=q['desired'];count=levels.count(desired)
 action='review' if q['flag'] and count>=q['threshold'] else 'routine'
 if family=='compare':return {'relation':relation(levels[0],levels[-1])}
 if family=='count':return {'count':count}
 if family=='select':return {'records':[chr(65+i) for i,v in enumerate(levels) if v==desired]}
 if family=='policy':return {'action':action}
 return {'first_vs_last':relation(levels[0],levels[-1]),'selected_count':count,'action':action}
def stat_questions(group):
 rs=group['records'];rng=rngkey(group['group_id']);field=rng.choice(FIELDS);desired=rng.choice(VALUES);flag=bool(rng.randrange(2));threshold=rng.randrange(1,len(rs)+1)
 levels=[r['bins'][field] for r in rs];names=[chr(65+i) for i in range(len(rs))]
 common='Read the '+field+' levels for EEG records '+', '.join(names)+'. levels must be an array in that record order, using low, mid or high. '
 result=[]
 for family in ['compare','count','select','policy','report']:
  q={'family':family,'field':field,'desired':desired,'flag':flag,'threshold':threshold}
  if family=='compare':
   text=common+'Compare record A with record '+names[-1]+'. Output only JSON with levels and relation. relation is lower, equal or higher for A relative to '+names[-1]+'.'
  elif family=='count':
   text=common+'Count how many records have level '+desired+'. Output only JSON with levels and count (an integer).'
  elif family=='select':
   text=common+'Select records with level '+desired+'. Output only JSON with levels and records (an ordered array of record letters, empty when none match).'
  elif family=='policy':
   text=common+'External workflow flag enabled='+str(flag).lower()+'. Return review only if enabled is true AND at least '+str(threshold)+' records have level '+desired+'; otherwise routine. Output only JSON with levels and action.'
  else:
   text=common+'Write a compact factual report. External flag enabled='+str(flag).lower()+'. Select records whose level is '+desired+'. Use action review only if enabled AND the selected count is at least '+str(threshold)+', otherwise routine. Output JSON with levels, first_vs_last (lower/equal/higher for A relative to '+names[-1]+'), selected_count (integer), action and summary (one short sentence about the observations, no diagnosis).'
  q.update(question=text,answer={'levels':levels,**derive(levels,q)})
  result.append(q)
 return result
def parse_stat(text,q):
 try:v=json.loads(text.strip())
 except (ValueError,TypeError):return None,False,0.,0.,False,False,None
 if not isinstance(v,dict):return v,False,0.,0.,False,False,None
 expected=q['answer'];keys=set(expected)|({'summary'} if q['family']=='report' else set())
 levels=v.get('levels');levels_ok=isinstance(levels,list) and len(levels)==len(expected['levels']) and all(x in VALUES for x in levels)
 valid=set(v)==keys and levels_ok
 if q['family']=='report':valid=valid and isinstance(v.get('summary'),str) and 0<len(v['summary'])<=2000
 for k,x in expected.items():
  if k=='levels':continue
  y=v.get(k);valid=valid and type(y)==type(x)
  if k=='records':valid=valid and all(isinstance(a,str) and a in [chr(65+i) for i in range(len(expected['levels']))] for a in y)
 fact=mean((levels[i] if isinstance(levels,list) and i<len(levels) else None)==x for i,x in enumerate(expected['levels']))
 decision={k:v.get(k) for k in expected if k!='levels'};gold={k:x for k,x in expected.items() if k!='levels'}
 answer_ok=decision==gold
 consistent=decision==derive(levels,q) if levels_ok else False
 allfacts=fact==1.
 exact=bool(valid and allfacts and answer_ok)
 return v,bool(valid),fact,float(answer_ok),allfacts,exact,bool(consistent) if levels_ok else None
def sleep_segment(rec,length):
 n=len(rec['stage']);count=n if length=='whole' else min(int(length),n)
 start=0 if length=='whole' else int(hashlib.sha256((str(SEED)+rec['subject']+str(count)).encode()).hexdigest()[:12],16)%(n-count+1)
 return {'subject':rec['subject'],'start_epoch':start,'start_seconds':start*30,'stage':rec['stage'][start:start+count],'n':count,'requested_length':length}
def sleep_report(stages):
 counts=collections.Counter(stages);dominant=max(range(6),key=lambda i:(counts[i],-i));known=sum(x!=5 for x in stages)
 sleep=sum(x in [1,2,3,4] for x in stages)/max(1,known)
 return {'first_stage':STAGES[stages[0]],'last_stage':STAGES[stages[-1]],'dominant_stage':STAGES[dominant],'sleep_share':'unknown' if known==0 else 'low' if sleep<.25 else 'mid' if sleep<.6 else 'high'}
def sleep_questions(seg):
 n=seg['n'];rng=rngkey(seg['subject']+str(seg['start_epoch'])+str(n));points=[0,n//2,n-1]
 stages=[STAGES[seg['stage'][i]] for i in points];i=rng.randrange(n-1);pair=[STAGES[seg['stage'][i]],STAGES[seg['stage'][i+1]]]
 changed=pair[0]!=pair[1];flag=bool(rng.randrange(2))
 return [
 {'family':'time_trace','indices':points,'question':'Read sleep stages at relative epoch indices '+','.join(map(str,points))+'. Return only JSON with stages, an array in exactly that index order.','answer':{'stages':stages}},
 {'family':'adjacent_transition','indices':[i,i+1],'question':'Read stages at relative epoch indices '+str(i)+' and '+str(i+1)+'. Is the stage different between those epochs? Return only JSON with stages (two strings in order) and changed (yes or no).','answer':{'stages':pair,'changed':'yes' if changed else 'no'}},
 {'family':'text_policy','indices':[i,i+1],'flag':flag,'question':'External workflow enabled='+str(flag).lower()+'. Read stages at relative indices '+str(i)+' and '+str(i+1)+'. Action review if enabled AND those two stages differ, otherwise routine. Return only JSON with stages (two strings in order) and action. This policy is a synthetic workflow rule, not a medical recommendation.','answer':{'stages':pair,'action':'review' if flag and changed else 'routine'}},
 {'family':'report','indices':None,'question':'Write a short factual report for this observed EEG segment. Return JSON with first_stage,last_stage,dominant_stage,sleep_share,summary. Stages are wake,n1,n2,n3,rem,unknown. dominant_stage is the most frequent stage (ties use that listed order). sleep_share is the fraction of n1+n2+n3+rem among known epochs: low below0.25, mid from0.25 to below0.60, high otherwise (unknown if there are no known epochs). summary is one short sentence about these facts. Do not diagnose or describe unobserved time.','answer':sleep_report(seg['stage'])}
 ]
def parse_sleep(text,q):
 try:v=json.loads(text.strip())
 except (ValueError,TypeError):return None,False,0.,0.,False,False,None
 if not isinstance(v,dict):return v,False,0.,0.,False,False,None
 keys=set(q['answer'])|({'summary'} if q['family']=='report' else set());valid=set(v)==keys
 if q['family']=='report':
  valid=valid and all(v.get(k) in STAGES for k in ['first_stage','last_stage','dominant_stage']) and v.get('sleep_share') in VALUES+['unknown'] and isinstance(v.get('summary'),str) and 0<len(v['summary'])<=2000
  fact=mean(v.get(k)==x for k,x in q['answer'].items());return v,bool(valid),fact,None,fact==1.,bool(valid and fact==1.),None
 stages=v.get('stages');ok=isinstance(stages,list) and len(stages)==len(q['answer']['stages']) and all(x in STAGES for x in stages)
 valid=valid and ok;fact=mean((stages[i] if isinstance(stages,list) and i<len(stages) else None)==x for i,x in enumerate(q['answer']['stages']))
 if q['family']=='time_trace':return v,bool(valid),fact,None,fact==1.,bool(valid and fact==1.),None
 key='changed' if q['family']=='adjacent_transition' else 'action'
 valid=valid and v.get(key) in (['yes','no'] if key=='changed' else ['review','routine'])
 answer_ok=v.get(key)==q['answer'][key]
 predicted=('yes' if stages[0]!=stages[1] else 'no') if key=='changed' and ok else ('review' if q.get('flag') and stages[0]!=stages[1] else 'routine') if ok else None
 return v,bool(valid),fact,float(answer_ok),fact==1.,bool(valid and fact==1. and answer_ok),v.get(key)==predicted if ok else None
def stat_groups(split,count):
 recs=read(D/(split+'_records.json'));groups=collections.defaultdict(list)
 for i,r in enumerate(recs):groups[r['dataset']].append((i,r))
 out=[]
 for ds,items in sorted(groups.items()):
  rng=rngkey(ds+':groups:'+split)
  for j in range(count):
   n=2 if j%2==0 else 4;chosen=rng.sample(items,n)
   out.append({'group_id':ds+':'+split+':'+str(j),'dataset':ds,'indices':[i for i,r in chosen],'records':[r for i,r in chosen],'n':n})
 return out
def prepare_role(role,split):
 out=R/role;path=out/(split+'_plan.json')
 if path.exists():return read(path)
 if split=='test':assert (out/'test_freeze.json').exists()
 if role=='statistics':
  groups=stat_groups(split,4 if split=='val' else 8)
  if split=='smoke':raise AssertionError('Use subset of Val plan for Smoke')
  plan={'role':role,'split':split,'groups':groups,'questions':sum(len(stat_questions(x)) for x in groups)}
 else:
  pilot=read(SD/'pilot_subjects.json')[split]
  records=sorted([r for r in read(SD/(split+'.json')) if r['subject'] in pilot],key=lambda x:x['subject'])
  plan={'role':role,'split':split,'subjects':[r['subject'] for r in records],'segments':[sleep_segment(r,l) for r in records for l in ['32','128','512','whole']],'questions':len(records)*4*4}
 write(path,plan);return plan
def cpu_check():
 assert set(read(SD/'pilot_subjects.json')['val']).isdisjoint(read(SD/'pilot_subjects.json')['test'])
 count=0
 for n in [2,4]:
  for levels in itertools.product(VALUES,repeat=n):
   group={'group_id':'fixture'+''.join(levels),'records':[{'bins':dict.fromkeys(FIELDS,v)} for v in levels]}
   for q in stat_questions(group):
    expected=dict(q['answer'])
    if q['family']=='report':expected['summary']='Observed encoded levels.'
    pred,valid,fact,decision,allfacts,exact,consistent=parse_stat(json.dumps(expected),q)
    assert valid and exact and allfacts and (consistent is True);count+=1
 for seq in [[0]*32,[1,2,3,4,0,5]*6,[0,0,4,4]*8]:
  seg={'subject':'fixture','start_epoch':0,'n':len(seq),'stage':seq}
  for q in sleep_questions(seg):
   expected=dict(q['answer'])
   if q['family']=='report':expected['summary']='Observed segment facts.'
   parsed=parse_sleep(json.dumps(expected),q);assert parsed[1] and parsed[5]
 for role in ['statistics','sleep']:prepare_role(role,'val')
 write(R/'audit/cpu_checks.json',{'status':'passed','statistical_fixture_cases':count,'sleep_fixtures':12,'no_test_read_for_task_design':True,'source_sha256':sha(__file__)})
def metrics(rows):
 out={}
 for mode in sorted({r['mode'] for r in rows}):
  out[mode]={}
  for family in sorted({r['family'] for r in rows if r['mode']==mode}):
   rr=[r for r in rows if r['mode']==mode and r['family']==family];decisions=[r for r in rr if r['decision_score'] is not None];correctfacts=[r for r in decisions if r['all_facts_correct'] and r['valid']]
   consistent=[r['self_consistent'] for r in rr if r['self_consistent'] is not None]
   out[mode][family]={'queries':len(rr),'format_valid_rate':mean(r['valid'] for r in rr),'fact_accuracy':mean(r['fact_score'] for r in rr),'decision_accuracy':mean(r['decision_score'] if r['valid'] else 0. for r in decisions) if decisions else None,'decision_value_accuracy':mean(r['decision_score'] for r in decisions) if decisions else None,'exact_accuracy':mean(r['correct'] for r in rr),'all_facts_correct_rate':mean(r['all_facts_correct'] for r in rr),'correct_fact_support':len(correctfacts),'decision_accuracy_given_correct_facts':mean(r['decision_score'] for r in correctfacts) if correctfacts else None,'self_consistency_given_parseable_facts':mean(consistent) if consistent else None,'self_consistency_support':len(consistent)}
   out[mode][family]['by_scope']={}
   for scope in sorted({r.get('dataset',r.get('requested_length')) for r in rr}):
    xx=[r for r in rr if r.get('dataset',r.get('requested_length'))==scope]
    out[mode][family]['by_scope'][scope]={'queries':len(xx),'format_valid_rate':mean(r['valid'] for r in xx),'fact_accuracy':mean(r['fact_score'] for r in xx),'decision_accuracy':mean(r['decision_score'] if r['valid'] else 0. for r in xx if r['decision_score'] is not None) if any(r['decision_score'] is not None for r in xx) else None,'exact_accuracy':mean(r['correct'] for r in xx)}
 return {'status':'complete','queries':len(rows),'per_mode_family':out,'interpretation':'Frozen inference probe. Fact reading, gold-target decision, and consistency with generated facts separated. Conditional metrics may have tiny support; no training or clinical validity.'}
def load_runtime(role):
 import torch
 from transformers import AutoTokenizer,AutoModelForCausalLM
 import stat_joint28_1015 as joint
 import statsem_1014_train as base
 import stat_compact28_1015_v3 as compact
 import sleep_long_native_1014 as sleep
 torch.set_num_threads(4);random.seed(SEED);torch.manual_seed(SEED)
 budget=int(os.environ['VSTATS_COMPACT_ALLOCATOR_BUDGET_MIB']);total=int(os.environ['VSTATS_COMPACT_GPU_TOTAL_MIB'])
 torch.cuda.set_per_process_memory_fraction(min(1.,budget/total))
 tok=AutoTokenizer.from_pretrained(MODEL,local_files_only=True)
 lm=AutoModelForCausalLM.from_pretrained(MODEL,torch_dtype=torch.bfloat16,local_files_only=True).cpu()
 emb=copy.deepcopy(lm.get_input_embeddings());del lm
 if role=='statistics':
  net=joint.initialized().eval();net.load_state_dict(torch.load(J/'train/best.pt',map_location='cpu',weights_only=False)['state_dict']);net.requires_grad_(False)
  oracle=compact.load_compact('cuda').eval().requires_grad_(False)
 else:
  ck=torch.load(S/'oracle/best.pt',map_location='cpu',weights_only=False)['state_dict']
  oracle={k:v.cuda() for k,v in ck.items()}
  net=sleep.student_model().eval();net.load_state_dict(torch.load(S/'student/best.pt',map_location='cpu',weights_only=False)['state_dict']);net.requires_grad_(False)
 return torch,tok,emb,net,oracle,joint,base,sleep
def stat_prompt(tok,emb,group,q,prefix):
 import torch
 marker='__RECORDS__';template=tok.apply_chat_template([{'role':'user','content':marker}],tokenize=False,add_generation_prompt=True,enable_thinking=False);bef,aft=template.split(marker)
 def e(s):return emb(torch.tensor(tok.encode(s,add_special_tokens=False),dtype=torch.long)).detach()
 parts=[e(bef+'Observed EEG records. Each record has28 evidence tokens. Records are separate samples; their order does not imply elapsed time.\n')]
 for i,p in enumerate(prefix):parts.extend([e('Record '+chr(65+i)+':\n'),p.to(dtype=emb.weight.dtype,device='cpu'),e('\n')])
 parts.append(e(q['question']+aft));return torch.cat(parts)
def evaluate(role,split):
 import torch
 from vllm import LLM,SamplingParams
 guard();plan=prepare_role(role,'val' if split=='smoke' else split)
 out=R/role/'evaluation';out.mkdir(exist_ok=True);path=out/(split+'.jsonl')
 if path.with_suffix('.summary.json').exists():return
 if path.exists():
  failed=out/'failed_attempts';failed.mkdir(exist_ok=True);path.rename(failed/(path.name+'.'+str(int(time.time()))))
 torch,tok,emb,net,oracle,joint,base,sleep=load_runtime(role)
 arbatch=int(os.environ.get('VSTATS_COMPACT_AR_BATCH','4')) if role=='statistics' else 1
 maxlen=4096 if role=='statistics' else 32768
 engine=LLM(model=str(MODEL),dtype='bfloat16',enable_prompt_embeds=True,max_model_len=maxlen,max_num_seqs=arbatch,gpu_memory_utilization=float(os.environ.get('VSTATS_COMPACT_AR_MEMORY_UTILIZATION','.35')),enforce_eager=True,disable_log_stats=True,seed=SEED)
 params=SamplingParams(temperature=0.,max_tokens=384);rows=[];began=time.time()
 modes=['oracle','eeg'] if role=='sleep' or split=='smoke' else ['oracle','eeg','mismatched_eeg']
 if role=='statistics':
  logicalsplit='val' if split=='smoke' else split
  old=joint.data(logicalsplit);bank=joint.banks(logicalsplit)
  groups=plan['groups'][:2] if split=='smoke' else plan['groups']
  pref_by_group={}
  with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
   for group in groups:
    ii=group['indices'];rs=group['records']
    real=net(old['bank'][ii].cuda(),old['valid'][ii].cuda(),bank['bank'][ii].cuda(),bank['valid'][ii].cuda()).cpu()
    bb,ll=base.teacher_batch(rs,'cuda');gold=oracle(bb,ll).cpu()
    pref_by_group[group['group_id']]={'oracle':gold,'eeg':real}
  # Deterministic within-dataset matched-size derangement; target group IDs never change.
  for group in groups:
   peers=[x for x in groups if x['dataset']==group['dataset'] and x['n']==group['n']]
   if len(peers)>1:
    donor=peers[(peers.index(group)+1)%len(peers)];group['mismatched_input_ids']=[r['id'] for r in donor['records']];pref_by_group[group['group_id']]['mismatched_eeg']=pref_by_group[donor['group_id']]['eeg']
   else:group['mismatched_input_ids']=[r['id'] for r in group['records'][::-1]];pref_by_group[group['group_id']]['mismatched_eeg']=pref_by_group[group['group_id']]['eeg'].flip(0)
  pairs=[(mode,g,q) for mode in modes for g in groups for q in stat_questions(g)]
  with path.open('w') as f:
   for off in range(0,len(pairs),arbatch):
    batch=pairs[off:off+arbatch];requests=[]
    for mode,g,q in batch:
     x=stat_prompt(tok,emb,g,q,pref_by_group[g['group_id']][mode]);assert len(x)+384<=maxlen
     requests.append({'prompt_embeds':x.bfloat16()})
    outputs=engine.generate(requests,params,use_tqdm=False)
    for (mode,g,q),o,request in zip(batch,outputs,requests):
     text=o.outputs[0].text;pred,valid,fact,decision,allfacts,exact,consistent=parse_stat(text,q)
     row={'role':role,'split':split,'mode':mode,'group_id':g['group_id'],'dataset':g['dataset'],'sample_ids':[r['id'] for r in g['records']],'input_sample_ids':g['mismatched_input_ids'] if mode=='mismatched_eeg' else [r['id'] for r in g['records']],'records':g['n'],'soft_tokens':28*g['n'],'family':q['family'],'field':q['field'],'question':q['question'],'task_spec':q,'expected':q['answer'],'prediction':pred,'generation':text,'valid':valid,'fact_score':fact,'decision_score':decision,'all_facts_correct':allfacts,'correct':exact,'self_consistent':consistent,'prompt_tokens':len(request['prompt_embeds']),'gold_input':mode=='oracle'}
     rows.append(row);f.write(json.dumps(row,ensure_ascii=False)+'\n')
    f.flush()
    if off%40==0:print('AR',role,split,off,len(pairs),flush=True)
 else:
  segments=[plan['segments'][0]] if split=='smoke' else plan['segments']
  with path.open('w') as f:
   for mode in modes:
    for seg in segments:
     with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
      if mode=='oracle':pref=(oracle['codes'][torch.tensor(seg['stage'],device='cuda')]+oracle['position']).cpu()
      else:
       cached=torch.load(S/'eeg_cache'/(seg['subject']+'.pt'),map_location='cpu',weights_only=False)
       assert cached['metadata']['subject']==seg['subject'] and not cached['metadata']['stage_input']
       a=seg['start_epoch'];bank=cached['bank'][a:a+seg['n']];assert len(bank)==seg['n']
       pref=(sleep.student_prefix(net,bank.cuda())+oracle['position']).cpu()
     for q in sleep_questions(seg):
      x=sleep.prompt(tok,emb,seg,q,pref);assert len(x)+384<=maxlen,(len(x),seg['n'])
      text=engine.generate([{'prompt_embeds':x.bfloat16()}],params,use_tqdm=False)[0].outputs[0].text
      pred,valid,fact,decision,allfacts,exact,consistent=parse_sleep(text,q)
      row={'role':role,'split':split,'mode':mode,'subject':seg['subject'],'start_epoch':seg['start_epoch'],'start_seconds':seg['start_seconds'],'epochs':seg['n'],'requested_length':seg['requested_length'],'soft_tokens':16*seg['n'],'family':q['family'],'question':q['question'],'task_spec':q,'expected':q['answer'],'prediction':pred,'generation':text,'valid':valid,'fact_score':fact,'decision_score':decision,'all_facts_correct':allfacts,'correct':exact,'self_consistent':consistent,'prompt_tokens':len(x),'gold_input':mode=='oracle','scope':'Original HMC6subject pilot; continuous30sepochs; frozen native32 interface extrapolation; not a newly trained long-context model'}
      rows.append(row);f.write(json.dumps(row,ensure_ascii=False)+'\n');f.flush()
     print('AR',role,split,mode,seg['subject'],seg['requested_length'],len(rows),flush=True)
 result=metrics(rows);result.update(role=role,split=split,source_sha256=sha(__file__),seconds=time.time()-began,temperature=0.,max_new_tokens=384,peak_torch_gib=torch.cuda.max_memory_allocated()/2**30)
 write(path.with_suffix('.summary.json'),result);write(path.with_suffix('.failures.json'),[r for r in rows if not r['correct']][:60])
def verify():
 conf=read(R/'protocol.json');assert sha(__file__)==conf['source_sha256']
 for p,h in conf['dependencies'].items():assert sha(P/p)==h,p
 for p,h in conf['models'].items():assert sha(p)==h,p
def stage(role,split,deadline):
 W=R/('worker_'+role);statepath=W/(split+'.attempt_state.json')
 meta=read(statepath) if statepath.exists() else {'first_started':None,'attempts':0,'ar_batch':4 if role=='statistics' else 1,'ar_memory_utilization':.35}
 artifact=R/role/'evaluation'/(split+'.summary.json')
 if artifact.exists():return
 while True:
  limit=min(deadline,meta['first_started']+4*3600 if meta['first_started'] else deadline)
  if time.time()>=limit:raise TimeoutError('Original worker/stage budget reached')
  guard();lease=None;env=dict(os.environ)
  while lease is None:
   cards=[]
   q=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.total,memory.free,utilization.gpu','--format=csv,noheader,nounits'],text=True)
   for line in q.splitlines():
    gpu,total,free,util=[int(x.strip()) for x in line.split(',')]
    if gpu in range(2,8) and free>=24*1024:cards.append((gpu,total,free,util))
   for gpu,total,free,util in sorted(cards,key=lambda x:(-x[2],x[3])):
    f=(R/('gpu'+str(gpu)+'.lease')).open('a+')
    try:fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError:f.close();continue
    free=int(subprocess.check_output(['nvidia-smi','-i',str(gpu),'--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True).strip())
    if free<24*1024:fcntl.flock(f,fcntl.LOCK_UN);f.close();continue
    lease=f;env.update(CUDA_VISIBLE_DEVICES=str(gpu),VSTATS_COMPACT_ALLOCATOR_BUDGET_MIB=str(free-10*1024),VSTATS_COMPACT_GPU_TOTAL_MIB=str(total),VSTATS_COMPACT_AR_BATCH=str(meta['ar_batch']),VSTATS_COMPACT_AR_MEMORY_UTILIZATION=str(meta['ar_memory_utilization']));break
   if lease is None:
    write(W/'status.json',{'state':'waiting_gpu_memory','stage':split,'time':time.time(),'pipeline_pid':os.getpid(),'minimum_free_gib':24})
    if time.time()>=limit:raise TimeoutError('Waiting VRAM within original budget')
    time.sleep(10)
  try:
   if meta['first_started'] is None:meta['first_started']=time.time()
   meta['attempts']+=1;write(statepath,meta)
   log=W/(split+'.log');offset=log.stat().st_size if log.exists() else 0
   cmd=[ARPY,'-B',str(Path(__file__).resolve()),'eval','--role',role,'--split',split]
   with log.open('a') as f:
    f.write(json.dumps({'time':time.time(),'attempt':meta,'gpu':gpu,'source_sha256':sha(__file__)})+'\n');f.flush()
    child=subprocess.Popen(cmd,cwd=P,env=env,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
    write(W/'status.json',{'state':'running','stage':split,'pid':child.pid,'pipeline_pid':os.getpid(),'gpu':gpu,'time':time.time(),'command':cmd,'log':str(log)})
    try:code=child.wait(timeout=max(1,min(deadline,meta['first_started']+4*3600)-time.time()))
    except subprocess.TimeoutExpired:
     os.killpg(child.pid,signal.SIGTERM)
     try:child.wait(timeout=30)
     except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()
     raise TimeoutError('Original stage budget expired')
   if code==0:assert artifact.exists();return
   with log.open() as f:f.seek(offset);tail=f.read().lower()
   if not any(x in tail for x in ['cuda out of memory','torch.outofmemoryerror','cuda error: out of memory']):raise RuntimeError(role+' '+split+' failed; see '+str(log))
   before=dict(meta)
   if meta['ar_batch']==1 and meta['ar_memory_utilization']<=.2:raise RuntimeError('OOM at minimum allowed resources')
   meta['ar_batch']=max(1,meta['ar_batch']//2);meta['ar_memory_utilization']=max(.2,round(meta['ar_memory_utilization']-.05,2));write(statepath,meta)
   with (W/'resource_retries.jsonl').open('a') as f:f.write(json.dumps({'time':time.time(),'before':before,'after':meta,'failed_pid':child.pid,'same_models_tasks_budget':True})+'\n')
  finally:
   fcntl.flock(lease,fcntl.LOCK_UN);lease.close()
def pipeline(role):
 W=R/('worker_'+role);W.mkdir(exist_ok=True)
 with (W/'pipeline.lock').open('a+') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  first=W/'first_launch.json'
  if not first.exists():write(first,{'started':time.time()})
  started=read(first)['started'];deadline=started+12*3600
  write(W/'launch.json',{'pid':os.getpid(),'started':started,'source_sha256':sha(__file__)})
  try:
   verify();stage(role,'smoke',deadline);stage(role,'val',deadline)
   freeze=R/role/'test_freeze.json'
   if not freeze.exists():
    verify();write(freeze,{'status':'frozen','time':time.time(),'source_sha256':sha(__file__),'models':read(R/'protocol.json')['models'],'selection':'None. Existing checkpoints frozen before this probe; Val is runtime and descriptive only; no fitting/selection by Val/Test.'})
   stage(role,'test',deadline)
   write(W/'complete.json',{'status':'complete','time':time.time(),'role':role})
   write(W/'status.json',{'state':'complete','time':time.time(),'role':role})
  except BaseException:
   write(W/'status.json',{'state':'failed','time':time.time(),'role':role,'traceback':traceback.format_exc()});raise
def main():
 parser=argparse.ArgumentParser();parser.add_argument('action',choices=['cpu','pipeline','eval']);parser.add_argument('--role',choices=['statistics','sleep']);parser.add_argument('--split',choices=['smoke','val','test']);args=parser.parse_args()
 if args.action=='cpu':cpu_check()
 elif args.action=='pipeline':pipeline(args.role)
 else:evaluate(args.role,args.split)
if __name__=='__main__':main()
