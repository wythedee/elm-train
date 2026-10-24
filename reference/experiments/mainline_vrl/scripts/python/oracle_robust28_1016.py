#!/usr/bin/env python3
"""Train only the factorized 28-token Oracle with a wider, fixed task distribution."""

if __name__ == "__main__":
    raise SystemExit("Reference snapshot only. Use tools/workbench.py for inspection; configure a fresh run before training.")

from pathlib import Path
import argparse,ast,collections,copy,csv,fcntl,hashlib,json,math,os,random,shutil,signal,subprocess,sys,time,traceback
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
import stat_compact28_1015_v3 as ref
import stat_joint28_1015 as joint
import reason_sleep_1015 as probe
base=ref.base;P=ref.P;CR=ref.R;DATA=ref.DATA
R=P/'experiments/mainline_vrl/runs/oracle_robust28_20261016_s1014'
PR=probe.R
FIELDS=base.FIELDS;VALUES=base.VALUES;SEED=1014
TRAINPY=ref.TRAINPY;ARPY=ref.ARPY
read=ref.read;write=ref.write;sha=ref.sha;complete=ref.complete
FRAMES={
 'train':['Respond to the requested observation task.','Use only the supplied observations and the stated rule.'],
 'val':['Answer this observation query in the required JSON schema.','Resolve the request using the evidence attached to each record.'],
 'test':['Produce the requested structured result from the recorded facts.','Follow this query and report only the fields it requests.']}
BIN_WORDS={
 'train':['Use low, mid or high for each level.','The three ordered levels are low < mid < high.'],
 'val':['Levels are ordered from low through mid to high.','Represent each measurement by low, mid or high.'],
 'test':['Encode the measurement tier with low, mid or high.','The ordinal scale is low, then mid, then high.']}
def guard(extra=0):
 if shutil.disk_usage(R).free<12*2**30+extra:raise OSError('12GiB disk/atomic-write reserve reached')
def save(p,x):
 guard(128*2**20);base.save_checkpoint(Path(p),x)
def seed():base.seed_all()
def context(rng,split):
 notes={
 'train':['The export filename is an administrative identifier.','The viewer accepts structured observations.','A scheduling entry does not describe the signal.','Archive this record after the requested output.'],
 'val':['A coordinator is preparing the observation register.','Folder names carry no measurement values.','The attached form has no additional physiological evidence.'],
 'test':['The case index is used only for document retrieval.','Administrative dates are unrelated to measured levels.','The destination folder does not change any observation.']}
 n=rng.choice([0,4,8,16,24])
 return '\n'.join('Administrative note '+str(i)+': '+rng.choice(notes[split]) for i in range(n))
def actual_values(r):return dict(zip(FIELDS,[VALUES[i] for i in base.bin_values(r)]))
def augment(q,rng,split):
 q=copy.deepcopy(q);frame=rng.choice(FRAMES[split]);ctx=context(rng,split)
 if q['family'] in ['paraphrase','context']:
  field=q.get('field') or (q['task'].split(':',1)[1] if q['task'].startswith('feature:') else 'label')
  if field=='label':
   qs={'train':['Return the observed class as a JSON object with only label.','Extract the class string; output exactly {"label":"CLASS"}.'],
       'val':['Which observed class is encoded? Reply with only the label JSON key.','Read the class observation and serialize it under label.'],
       'test':['Report the class carried by this evidence using the sole JSON key label.','Retrieve the observed class, expressed as {"label":"CLASS"}.']}
  else:
   qs={'train':['Return only the '+field+' observation under that exact JSON key.','Read '+field+' and produce a one-key JSON object named '+field+'.'],
       'val':['What tier is encoded for '+field+'? Use only the JSON key '+field+'.','Retrieve '+field+' from the evidence, with its name as the sole JSON key.'],
       'test':['Report the encoded '+field+' tier in a JSON object with no other keys.','Extract '+field+' and serialize the value under its own name.']}
  q['question']=rng.choice(qs[split])+' '+rng.choice(BIN_WORDS[split]) if field!='label' else rng.choice(qs[split])
  if q['family']=='context' and not ctx:ctx=context(random.Random(rng.randrange(2**32)),split) or 'Administrative record. This note contains no signal measurements.'
 q['question']=frame+'\n'+((ctx+'\n') if ctx else '')+q['question']+'\nReturn JSON only, without a Markdown fence or explanation.'
 return q
def multi_q(group,family,rng,split):
 q=copy.deepcopy(next(q for q in probe.stat_questions(group) if q['family']==family))
 field=q['field'];names=[chr(65+i) for i in range(group['n'])];last=names[-1]
 starters={
 'train':['Read '+field+' for records '+', '.join(names)+'.','Extract the '+field+' tiers of '+', '.join(names)+'.'],
 'val':['Retrieve the '+field+' level separately for '+', '.join(names)+'.','For each listed record ('+', '.join(names)+'), read '+field+'.'],
 'test':['Decode '+field+' in the record sequence '+', '.join(names)+'.','Find the '+field+' observation for every record: '+', '.join(names)+'.']}
 common=rng.choice(starters[split])+' Return levels as an array in that exact record order. '+rng.choice(BIN_WORDS[split])+' '
 if family=='compare':
  body='Compare A to '+last+' on this ordered scale. Return JSON with levels and relation, where relation is lower, equal or higher for A relative to '+last+'.'
 elif family=='count':
  body='Count the records whose level is '+q['desired']+'. Return JSON with levels and count; count must be an integer.'
 elif family=='select':
  body='Select only the records at level '+q['desired']+'. Return JSON with levels and records; records is an ordered array of matching record letters, or [] if none match.'
 elif family=='policy':
  body='External workflow flag enabled='+str(q['flag']).lower()+'. Choose review if enabled is true AND the number of '+q['desired']+' records is at least '+str(q['threshold'])+'; choose routine otherwise. Return JSON with levels and action.'
 else:
  body='Prepare a concise factual report. External workflow flag enabled='+str(q['flag']).lower()+'. Count records at level '+q['desired']+'. action is review exactly when enabled is true AND this count is at least '+str(q['threshold'])+', and routine otherwise. Return JSON with levels, first_vs_last (lower/equal/higher for A relative to '+last+'), selected_count (integer), action, and summary (one short factual sentence, no diagnosis).'
 ctx=context(rng,split)
 q['question']=rng.choice(FRAMES[split])+'\n'+((ctx+'\n') if ctx else '')+common+body+'\nOutput JSON only, no Markdown or extra keys.'
 return q
def example(records,q,kind,scope,ident):
 return {'id':ident,'records':records,'q':q,'kind':kind,'scope':scope,'dataset':records[0]['dataset']}
def draw(rng,split,i,forced=None):
 labels=read(R/'protocol.json')['labels']
 u=rng.random();kind=forced or ('base' if u<.4 else 'single' if u<.7 else 'multi')
 count=rng.choice([2,4]) if kind=='multi' else 1
 rs=[base.synthetic_record(rng,labels,split,i*4+j) for j in range(count)]
 for r in rs:r['bins']=actual_values(r)
 if kind=='base':q=base.question(rs[0],rng.choice(base.TASKS),rng)
 elif kind=='single':q=augment(rng.choice(joint.robust_questions(rs[0])),rng,split)
 else:
  g={'group_id':'robust28:'+split+':'+str(i),'records':rs,'n':count}
  q=multi_q(g,rng.choice(['compare','count','select','policy','report']),rng,split)
 return example(rs,q,kind,'synthetic','synthetic:'+split+':'+str(i))
def target(e):
 q=e['q'];a=copy.deepcopy(q['answer'])
 if e['kind']=='multi' and q['family']=='report':
  a['summary']='Recorded '+q['field']+' levels in order are '+', '.join(a['levels'])+'.'
 return json.dumps(a,separators=(',',':'))
def plan(split):
 path=R/(split+'_plan.json')
 if path.exists():return read(path)
 if split=='test':assert (R/'test_freeze.json').exists()
 rng=random.Random(SEED+(60000 if split=='val' else 90000));items=[]
 # Identical 96 synthetic examples support old/text/new paired controls.
 for i in range(96):
  kind=['base','single','multi'][i%3];e=draw(rng,split,i,kind)
  # Force systematic coverage of all canonical/robust/multi query kinds.
  rr=e['records'];j=i//3
  if kind=='base':e['q']=base.question(rr[0],base.TASKS[j%9],rng)
  elif kind=='single':e['q']=augment(joint.robust_questions(rr[0])[j%18],rng,split)
  else:
   g={'group_id':e['id'],'records':rr,'n':len(rr)}
   e['q']=multi_q(g,['compare','count','select','policy','report'][j%5],rng,split)
  items.append(e)
 rr=read(DATA/(split+'_records.json'));indices=read(CR/'llm_new'/(split+'_indices.json'))
 # Fixed original selected Val256 / exposed Test2048, no split or ID changes.
 for i in indices:
  for j,q in enumerate(base.fixed_questions(rr[i])):items.append(example([rr[i]],q,'base','original_base',rr[i]['id']+':base:'+str(j)))
 # Same per-domain SHA-chosen robustness ID policy, with held-out wording.
 robust=[]
 for ds in sorted({r['dataset'] for r in rr}):
  ii=[i for i in indices if rr[i]['dataset']==ds]
  ii.sort(key=lambda i:hashlib.sha256((rr[i]['id']+':robust28').encode()).hexdigest())
  robust+=ii[:(2 if split=='val' else 8)]
 for i in robust:
  for j,q in enumerate(joint.robust_questions(rr[i])):
   items.append(example([rr[i]],augment(q,rng,split),'single','robust_single',rr[i]['id']+':robust:'+str(j)))
 groups=read(PR/'statistics'/(split+'_plan.json'))['groups']
 if split=='val':
  groups=[g for g in groups if int(g['group_id'].rsplit(':',1)[1])<2]
 for g in groups:
  for fam in ['compare','count','select','policy','report']:
   items.append(example(g['records'],multi_q(g,fam,rng,split),'multi','real_multi',g['group_id']+':'+fam))
 write(path,items);return items
def prepare():
 R.mkdir(parents=True,exist_ok=True);seed();guard()
 if not (R/'oracle_init.pt').exists():
  net=ref.load_compact('cpu')
  save(R/'oracle_init.pt',{'state_dict':net.state_dict(),'source_oracle_sha256':sha(CR/'oracle/best.pt'),'lineage':'Warm-start completed compact28 Oracle; new optimizer for augmentation phase, independent Smoke.'})
 net=initialized('cpu');bb=torch.tensor([[0,1,2,0,1,2]]);ll=torch.tensor([0]);pf=net(bb,ll)
 assert pf.shape==(1,28,2048)
 for j in range(6):
  changed=bb.clone();changed[0,j]=(changed[0,j]+1)%3;other=net(changed,ll)
  keep=torch.ones(28,dtype=torch.bool);keep[16+2*j:18+2*j]=False
  assert torch.equal(pf[:,keep],other[:,keep]) and not torch.equal(pf[:,~keep],other[:,~keep])
 assert torch.equal(pf[:,16:],net(bb,torch.tensor([1]))[:,16:])
 val=plan('val');rng=random.Random(SEED+98765)
 tf=[draw(rng,'val',i) for i in range(240)];write(R/'tf_val.json',tf)
 # Truth/parser checks span all 729 ordinal combinations for 2 and 4 records.
 checked=0
 for n in [2,4]:
  import itertools
  for vv in itertools.product(VALUES,repeat=n):
   for desired in VALUES:
    for flag in [False,True]:
     for threshold in [1,n]:
      q={'desired':desired,'flag':flag,'threshold':threshold}
      for fam in ['compare','count','select','policy','report']:
       q['family']=fam
       out=probe.derive(list(vv),q)
       rank={'low':0,'mid':1,'high':2};c=sum(x==desired for x in vv)
       rel='equal' if vv[0]==vv[-1] else ('lower' if rank[vv[0]]<rank[vv[-1]] else 'higher')
       act='review' if flag and c>=threshold else 'routine'
       expected={'compare':{'relation':rel},'count':{'count':c},'select':{'records':[chr(65+i) for i,x in enumerate(vv) if x==desired]},'policy':{'action':act},'report':{'first_vs_last':rel,'selected_count':c,'action':act}}[fam]
       assert out==expected,(vv,q,out,expected);checked+=1
 for e in val+tf:
  pred,valid,fact,decision,exact=parse(target(e),e)
  assert valid and fact==1 and exact,(e,pred)
  if all(r.get('synthetic') for r in e['records']):
   assert all(base.combo_partition(base.bin_values(r))==0 for r in e['records'])
  assert all(r['split']=='val' for r in e['records'])
 write(R/'audit/cpu_check.json',{'status':'passed','truth_fixture_checks':checked,'val_queries':len(val),'tf_queries':len(tf),'target_roundtrip':True,'val_bin_partition':0,'train_only_partitions':[2,3,4],'split_prompt_banks_disjoint':all(set(FRAMES[a]).isdisjoint(FRAMES[b]) for a,b in [('train','val'),('train','test'),('val','test')]),'no_test_access':True})
 write(R/'prepared.json',{'status':'complete','time':time.time(),'init_sha256':sha(R/'oracle_init.pt'),'val_plan_sha256':sha(R/'val_plan.json'),'tf_plan_sha256':sha(R/'tf_val.json')})
def initialized(device):
 net=ref.load_compact(device);net.load_state_dict(torch.load(R/'oracle_init.pt',map_location='cpu',weights_only=False)['state_dict'])
 return net
def render(tok,emb,e,prefix,text_mode=False):
 def enc(t):return emb(torch.tensor(base.token_ids(tok,t),device=emb.weight.device,dtype=torch.long)).detach()
 bef,aft=base.wrapper(tok);parts=[enc(bef)]
 if e['kind']!='multi':
  parts.append(enc('EEG evidence sample 1:\n'))
  r=e['records'][0]
  parts.append(enc(json.dumps({'label':r['label'],'statistics':actual_values(r)},separators=(',',':'))) if text_mode else prefix[0].to(emb.weight.device,dtype=emb.weight.dtype))
 else:
  parts.append(enc('EEG observations for independent records. Their order does not imply elapsed time.\n'))
  for j,r in enumerate(e['records']):
   parts.append(enc('Record '+chr(65+j)+':\n'))
   parts.append(enc(json.dumps({'label':r['label'],'statistics':actual_values(r)},separators=(',',':'))) if text_mode else prefix[j].to(emb.weight.device,dtype=emb.weight.dtype))
   parts.append(enc('\n'))
 parts.append(enc('\n'+e['q']['question']+aft));return torch.cat(parts)
def loss(lm,tok,emb,net,examples):
 sequences=[];labels=[]
 for e in examples:
  bb,ll=base.teacher_batch(e['records'],'cuda');pf=net(bb,ll)
  x=render(tok,emb,e,pf);t=torch.tensor(base.token_ids(tok,target(e),eos=True),device=x.device,dtype=torch.long)
  sequences.append(torch.cat([x,emb(t).detach()]));labels.append(torch.cat([torch.full((len(x),),-100,device=x.device,dtype=torch.long),t]))
 xx=nn.utils.rnn.pad_sequence(sequences,batch_first=True);yy=nn.utils.rnn.pad_sequence(labels,batch_first=True,padding_value=-100)
 length=torch.tensor([len(x) for x in sequences],device=xx.device);mask=torch.arange(len(xx[0]),device=xx.device)[None]<length[:,None]
 logits=lm(inputs_embeds=xx,attention_mask=mask.long(),use_cache=False).logits[:,:-1].float()
 targets=yy[:,1:];ok=targets!=-100;safe=targets.masked_fill(~ok,0)
 logp=F.log_softmax(logits,-1).gather(-1,safe[:,:,None]).squeeze(-1)*ok
 return -(logp.sum(-1)/ok.sum(-1).clamp_min(1)).mean()
def oracle_train(smoke=False):
 seed();out=R/('smoke' if smoke else 'oracle');out.mkdir(exist_ok=True)
 tok,lm,emb=base.hf_decoder();net=initialized('cuda');opt=torch.optim.AdamW(net.parameters(),lr=.001,weight_decay=.0001)
 rng=random.Random(SEED);steps=30 if smoke else 6000;micro=int(os.environ.get('VSTATS_COMPACT_MICRO_BATCH','2'))
 start=0;best=math.inf;history=[];began=time.time()
 if (out/'last.pt').exists():
  ck=torch.load(out/'last.pt',map_location='cpu',weights_only=False)
  net.load_state_dict(ck['state_dict']);opt.load_state_dict(ck['optimizer']);ref.restore(ck,rng);random.setstate(ck['python_global_rng'])
  start=ck['step'];best=ck['best'];history=ck['history']
 initial={k:p.detach().clone() for k,p in net.named_parameters()}
 def state(step):
  return {'state_dict':{k:v.detach().cpu() for k,v in net.state_dict().items()},'optimizer':opt.state_dict(),'step':step,'best':best,'history':history,**ref.rng_state(rng),'python_global_rng':random.getstate(),'source_sha256':sha(__file__),'tokens':28,'decoder_frozen':True,'effective_batch':8,'micro_batch':micro,'phase_initialization_sha256':sha(R/'oracle_init.pt')}
 vp=read(R/'tf_val.json')
 for step in range(start+1,steps+1):
  net.train();opt.zero_grad(set_to_none=True);ee=[draw(rng,'train',step*8+i) for i in range(8)];ce_value=0
  for off in range(0,8,micro):
   batch=ee[off:off+micro]
   with torch.autocast('cuda',dtype=torch.bfloat16):ce=loss(lm,tok,emb,net,batch);total=ce+net.anchor()
   assert torch.isfinite(total);(total*len(batch)/8).backward();ce_value+=float(ce)*len(batch)/8
  grad=torch.nn.utils.clip_grad_norm_(net.parameters(),1.);assert torch.isfinite(grad) and grad>0
  assert all(p.grad is None for p in lm.parameters());opt.step()
  if step%10==0:print(json.dumps({'step':step,'ce':ce_value,'grad':float(grad),'seconds':time.time()-began,'micro_batch':micro}),flush=True)
  if step%100==0 and not smoke:save(out/'last.pt',state(step))
  if smoke and step==steps:
   changed={k:not torch.equal(v,dict(net.named_parameters())[k]) for k,v in initial.items()}
   assert all(changed.values()),changed
   ck=state(step);save(out/'last.pt',ck)
   restored=initialized('cuda');restored.load_state_dict(torch.load(out/'last.pt',map_location='cpu',weights_only=False)['state_dict'])
   assert all(torch.equal(a,b) for a,b in zip(net.state_dict().values(),restored.state_dict().values()))
   write(out/'summary.json',{'status':'passed','steps':steps,'all_oracle_parameter_groups_changed':changed,'decoder_frozen':True,'checkpoint_roundtrip':True,'peak_torch_gib':torch.cuda.max_memory_allocated()/2**30,'seconds':time.time()-began})
  if not smoke and (step%500==0 or step==steps):
   net.eval();v=0
   with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
    for off in range(0,len(vp),micro):
     bb=vp[off:off+micro];v+=float(loss(lm,tok,emb,net,bb))*len(bb)/len(vp)
   improved=v<best;best=min(best,v);history.append({'step':step,'val_tf_loss':v,'seconds':time.time()-began})
   ck=state(step);save(out/'last.pt',ck)
   if improved:save(out/'best.pt',ck)
   write(out/'summary.json',{'status':'complete' if step==steps else 'running','steps':step,'best_val_tf':best,'history':history,'tokens':28,'trainable_parameters':sum(p.numel() for p in net.parameters()),'peak_torch_gib':torch.cuda.max_memory_allocated()/2**30,'seconds':time.time()-began})
   print('VALIDATION',history[-1],flush=True)
  if time.time()>float(os.environ.get('VSTATS_STAGE_DEADLINE','inf'))-90:
   save(out/'last.pt',state(step));raise TimeoutError('Cumulative stage budget reached; checkpoint retained')
def parse(text,e):
 q=e['q']
 if e['kind']=='multi':
  p,v,f,d,allfacts,exact,consistent=probe.parse_stat(text,q);return p,v,f,d,exact
 p,v,f=joint.parse(text,q);exact=v and p==q['answer']
 decision=float(exact) if q['task'].startswith('rule') else None
 return p,v,f,decision,exact
def summarize(rows):
 groups={}
 for r in rows:groups.setdefault((r['mode'],r['scope'],r['family']),[]).append(r)
 result=[]
 for (mode,scope,family),rr in sorted(groups.items()):
  ds={}
  for name in sorted({r['dataset'] for r in rr}):
   ss=[r for r in rr if r['dataset']==name];dec=[r for r in ss if r['decision_score'] is not None]
   ds[name]={'n':len(ss),'facts':float(np.mean([r['fact_score'] for r in ss])),'format':float(np.mean([r['valid'] for r in ss])),'exact':float(np.mean([r['correct'] for r in ss])),'decision':float(np.mean([r['decision_score'] if r['valid'] else 0 for r in dec])) if dec else None}
  value={'mode':mode,'scope':scope,'family':family,'n':len(rr),'per_dataset':ds}
  for k in ['facts','format','exact','decision']:
   vv=[v[k] for v in ds.values() if v[k] is not None];value[k]=float(np.mean(vv)) if vv else None
  result.append(value)
 canonical={}
 for mode in {r['mode'] for r in rows}:
  rr=[r for r in rows if r['mode']==mode and r['scope']=='original_base']
  if rr:canonical[mode]=base.metrics(rr)
 return {'status':'complete','queries':len(rows),'task_metrics':result,'canonical':canonical,'missing_tasks':'N/A, no composite score'}
def oracle_eval(scope):
 from transformers import AutoTokenizer,AutoModelForCausalLM
 from vllm import LLM,SamplingParams
 seed();guard();split='test' if scope=='test' else 'val';ee=plan(split)
 if scope=='preflight':ee=ee[:96];modes=['old','text']
 elif scope=='val':modes=['new']
 elif scope=='test':modes=['new']
 else:raise ValueError(scope)
 out=R/'evaluation';out.mkdir(exist_ok=True);path=out/(scope+'.jsonl');summary=out/(scope+'.summary.json')
 if summary.exists():return
 if path.exists():
  fail=out/'failed_attempts';fail.mkdir(exist_ok=True);path.rename(fail/(path.name+'.'+str(int(time.time()))))
 tok=AutoTokenizer.from_pretrained(base.MODEL,local_files_only=True)
 lm=AutoModelForCausalLM.from_pretrained(base.MODEL,torch_dtype=torch.bfloat16,local_files_only=True).cpu();emb=copy.deepcopy(lm.get_input_embeddings());del lm
 net=ref.load_compact('cuda').eval().requires_grad_(False)
 if 'new' in modes:net.load_state_dict(torch.load(R/'oracle/best.pt',map_location='cpu',weights_only=False)['state_dict'])
 batch=int(os.environ.get('VSTATS_COMPACT_AR_BATCH','8'))
 engine=LLM(model=str(base.MODEL),dtype='bfloat16',enable_prompt_embeds=True,max_model_len=4096,max_num_seqs=batch,gpu_memory_utilization=float(os.environ.get('VSTATS_COMPACT_AR_MEMORY_UTILIZATION','.35')),enforce_eager=True,disable_log_stats=True,seed=SEED)
 params=SamplingParams(temperature=0.,max_tokens=384);rows=[];began=time.time()
 pairs=[(mode,e) for mode in modes for e in ee]
 with path.open('w') as f:
  for off in range(0,len(pairs),batch):
   bb=pairs[off:off+batch];requests=[]
   with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
    for mode,e in bb:
     bins,labs=base.teacher_batch(e['records'],'cuda');pf=net(bins,labs).cpu()
     x=render(tok,emb,e,pf,mode=='text');assert len(x)+384<=4096
     requests.append({'prompt_embeds':x.bfloat16()})
   outputs=engine.generate(requests,params,use_tqdm=False)
   for (mode,e),o,request in zip(bb,outputs,requests):
    text=o.outputs[0].text;p,v,fact,d,exact=parse(text,e);q=e['q']
    row={'mode':mode,'scope':e['scope'],'id':e['id'],'dataset':e['dataset'],'sample_ids':[r['id'] for r in e['records']],'kind':e['kind'],'records':len(e['records']),'soft_tokens':0 if mode=='text' else 28*len(e['records']),'family':q.get('family',q.get('task')),'task':q.get('task',q.get('family')),'question':q['question'],'expected':q['answer'],'generation':text,'prediction':p,'valid':v,'fact_score':fact,'decision_score':d,'correct':bool(exact),'prompt_tokens':len(request['prompt_embeds']),'gold_label':e['records'][0]['label'],'gold_uses_labels_and_bins':True,'finish_reason':o.outputs[0].finish_reason}
    if 'field' in q:row['field']=q['field']
    rows.append(row);f.write(json.dumps(row,ensure_ascii=False)+'\n')
   f.flush()
   if off%80==0:print('AR',scope,off,len(pairs),flush=True)
 result=summarize(rows);result.update({'source_sha256':sha(__file__),'split':split,'seconds':time.time()-began,'peak_torch_gib':torch.cuda.max_memory_allocated()/2**30,'oracle_sha256':sha(R/'oracle/best.pt') if 'new' in modes else sha(CR/'oracle/best.pt'),'plan_sha256':sha(R/(split+'_plan.json')),'max_new_tokens':384,'temperature':0})
 write(summary,result);write(out/(scope+'.failures.json'),[r for r in rows if not r['correct']][:60])
def gate():
 d=read(R/'evaluation/val.summary.json');checks=[]
 for ds,v in d['canonical']['new']['per_dataset'].items():
  pass
 per=d['canonical']['new']['per_dataset']
 label=float(np.mean([v['tasks']['label']['bacc'] for v in per.values()]))
 fields={f:float(np.mean([v['tasks']['feature:'+f]['bacc'] for v in per.values()])) for f in FIELDS}
 checks.append({'name':'canonical_label_bacc','value':label,'minimum':.8})
 for f,v in fields.items():checks.append({'name':'canonical_'+f+'_bacc','value':v,'minimum':.8})
 checks.append({'name':'canonical_report_facts','value':float(np.mean([v['tasks']['report']['fact_accuracy'] for v in per.values()])),'minimum':.8})
 for v in d['task_metrics']:
  if v['scope'] not in ['robust_single','real_multi']:continue
  for k in ['facts','format']:
   checks.append({'name':v['scope']+':'+v['family']+':'+k,'value':v[k],'minimum':.8})
  if v['decision'] is not None:checks.append({'name':v['scope']+':'+v['family']+':strict_decision','value':v['decision'],'minimum':.8})
 passed=all(v['value']>=v['minimum'] for v in checks)
 write(R/'oracle_gate.json',{'status':'complete','passed':passed,'checks':checks,'failed':[v for v in checks if v['value']<v['minimum']],'val_only':True,'no_automatic_eeg_training':True})
 return passed
def verify():
 d=read(R/'protocol.json')
 assert sha(__file__)==d['source_sha256']
 for p,h in d['dependencies'].items():assert sha(P/p)==h,p
 assert sha(CR/'oracle/best.pt')==d['old_oracle_sha256']
 for split in ['train','val']:
  assert sha(DATA/(split+'_records.json'))==d['record_file_sha256'][split]
def report():
 paths=[p for p in (R/'evaluation').glob('*.summary.json') if read(p).get('status')=='complete']
 tables=[]
 for p in paths:
  for v in read(p)['task_metrics']:tables.append({'evaluation':p.stem,**{k:v[k] for k in ['mode','scope','family','n','facts','format','exact','decision']}})
 out=R/'details';out.mkdir(exist_ok=True)
 with (out/'tasks.csv').open('w') as f:
  writer=csv.DictWriter(f,fieldnames=['evaluation','mode','scope','family','n','facts','format','exact','decision']);writer.writeheader();writer.writerows(tables)
 lines=['# 28-token Oracle多任务训练','', '仅训练Oracle向量；冻结Qwen3-1.7B，不训练EEG。每样本分类16＋统计12；2/4独立记录不是连续时间。','',
 '从已完成compact28 Oracle warm-start，新增固定6000步。更多训练预算与覆盖同时变化，不作等预算因果归因。训练/Val/Test问法库分开；标签/统计合成独立，原组合切分与原数据subject split保留。','',
 '单样本原任务、换问法/上下文/输出结构、多样本事实与严格决策、格式分别报告。报告summary由程序事实构造，文字不评分，不是临床报告。正确文本只在96条配对探针实测，不定义所有LLM文本能力为100%。','',
 '|评测|输入|scope|任务|n|事实%|格式%|exact%|严格决策%|','|---|---|---|---|---:|---:|---:|---:|---:|']
 for v in tables:
  vals=[f"{100*v[k]:.2f}" if v[k] is not None else 'N/A' for k in ['facts','format','exact','decision']]
  lines.append('|'+ '|'.join([v['evaluation'],v['mode'],v['scope'],v['family'],str(v['n'])]+vals)+'|')
 g=read(R/'oracle_gate.json') if (R/'oracle_gate.json').exists() else None
 lines+=['','Val gate: '+json.dumps(g,ensure_ascii=False),'','Test若运行仍为已暴露的重复benchmark128/域，不是盲测或全量原Test。Oracle输入含gold，不是EEG能力；不声称保留原生推理能力。']
 (R/'final_report.md').write_text('\n'.join(lines)+'\n')
 write(R/'report_complete.json',{'status':'complete','time':time.time(),'report_sha256':sha(R/'final_report.md'),'gate_passed':g['passed'] if g else None})
 ledger=P/'experiments/mainline_vrl/reports/20260808_experiment_ledger.md'
 key='oracle_robust28_20261016_s1014 — Oracle多任务训练'
 if key not in ledger.read_text():
  with ledger.open('a') as f:f.write('\n\n### '+key+'\n\n仅训练28token Oracle，冻结LM；原split/组合分区/seed不变，独立Smoke后6000步。Val gate='+str(g['passed'] if g else None)+'；原任务/多记录/新问法/结构输出、正确文本探针分别见[报告](../runs/'+R.name+'/final_report.md)。独立诊断，不写旧分类榜；未训练EEG。\n')
def pipeline():
 W=R/'worker_main';W.mkdir(exist_ok=True)
 with (W/'pipeline.lock').open('a+') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);first=W/'first_launch.json'
  if not first.exists():write(first,{'started':time.time()})
  os.environ['VSTATS_STAT_EXPERT_DEADLINE']=str(read(first)['started']+12*3600)
  write(W/'launch.json',{'pid':os.getpid(),'started':read(first)['started'],'source_sha256':sha(__file__)})
  try:
   verify();stage('main','preflight',['oracle-eval','--scope','preflight'],'evaluation/preflight.summary.json')
   stage('main','oracle_smoke',['oracle-train','--smoke'],'smoke/summary.json')
   stage('main','oracle_training',['oracle-train'],'oracle/summary.json')
   stage('main','oracle_val',['oracle-eval','--scope','val'],'evaluation/val.summary.json')
   if gate():
    verify();write(R/'test_freeze.json',{'status':'frozen','time':time.time(),'oracle_sha256':sha(R/'oracle/best.pt'),'source_sha256':sha(__file__),'dependencies':read(R/'protocol.json')['dependencies'],'selection':'Best fixed Val TF across 6000-step training. No Test selection.'})
    stage('main','oracle_test',['oracle-eval','--scope','test'],'evaluation/test.summary.json')
   report()
   state='complete' if read(R/'oracle_gate.json')['passed'] else 'gate_failed'
   write(W/'status.json',{'state':state,'time':time.time(),'pipeline_pid':os.getpid()})
   write(W/'complete.json',{'status':'complete','state':state,'time':time.time()})
  except BaseException:
   write(W/'status.json',{'state':'failed','time':time.time(),'pipeline_pid':os.getpid(),'traceback':traceback.format_exc()});raise
def main():
 a=argparse.ArgumentParser();a.add_argument('command');a.add_argument('--smoke',action='store_true');a.add_argument('--scope',default='val');x=a.parse_args()
 if os.environ.get('CUDA_VISIBLE_DEVICES'):
  budget=int(os.environ['VSTATS_COMPACT_ALLOCATOR_BUDGET_MIB']);total=int(os.environ['VSTATS_COMPACT_GPU_TOTAL_MIB'])
  torch.cuda.set_per_process_memory_fraction(min(1.,budget/total))
 if x.command=='prepare':prepare()
 elif x.command=='pipeline':pipeline()
 elif x.command=='oracle-train':oracle_train(x.smoke)
 elif x.command=='oracle-eval':oracle_eval(x.scope)
 elif x.command=='report':report()
 else:raise ValueError(x.command)
# The stage launcher below is copied unchanged in semantics from the completed compact28:
# new-run logs/locks, 2-7 sharing, cumulative stage/worker deadlines, OOM-only microbatch retry.

def stage(role,name,args,artifact=None):
 root=R/'worker_main';root.mkdir(exist_ok=True);log=root/(name+'.log')
 if artifact and complete(R/artifact):print('SKIP',name,flush=True);return
 worker_deadline=float(os.environ['VSTATS_STAT_EXPERT_DEADLINE'])
 stage_meta=root/(name+'.attempt_state.json')
 meta=read(stage_meta) if stage_meta.exists() else {'first_started':None,'attempts':0,'micro_batch':2,'ar_batch':8,'ar_memory_utilization':.35}
 is_ar=args[0] in ['llm-eval','oracle-eval']
 is_gpu=is_ar or args[0] in ['oracle-train','llm-train']
 while True:
  deadline=min(worker_deadline,(meta['first_started']+4*3600) if meta['first_started'] is not None else worker_deadline)
  if time.time()>=deadline:raise TimeoutError('Original stage/worker budget reached before '+name)
  gpu_lock=None;env=dict(os.environ);chosen=None
  try:
   if is_gpu:
    need_mib=(24 if is_ar else 26)*1024
    while gpu_lock is None:
     q=subprocess.run(['nvidia-smi','--query-gpu=index,memory.total,memory.free,utilization.gpu','--format=csv,noheader,nounits'],capture_output=True,text=True,check=True)
     cards=[]
     for line in q.stdout.splitlines():
      gpu,total,free,util=[int(v.strip()) for v in line.split(',')]
      if gpu in range(2,8) and free>=need_mib:cards.append((gpu,total,free,util))
     cards.sort(key=lambda x:(-x[2],x[3],x[0]))
     for gpu,total,free,util in cards:
      f=(R/('gpu'+str(gpu)+'.lease')).open('a+')
      try:fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
      except BlockingIOError:f.close();continue
      # Recheck after acquiring our lease; other users are allowed to share.
      check=subprocess.run(['nvidia-smi','-i',str(gpu),'--query-gpu=memory.free','--format=csv,noheader,nounits'],capture_output=True,text=True,check=True)
      free=int(check.stdout.strip())
      if free>=need_mib:
       gpu_lock=f;chosen=str(gpu);env['CUDA_VISIBLE_DEVICES']=chosen
       # Leave8GiB physical reserve plus2GiB allowance for non-Torch allocations.
       env['VSTATS_COMPACT_ALLOCATOR_BUDGET_MIB']=str(free-10*1024)
       env['VSTATS_COMPACT_GPU_TOTAL_MIB']=str(total)
       break
      fcntl.flock(f,fcntl.LOCK_UN);f.close()
     if gpu_lock is None:
      write(root/'status.json',{'state':'waiting_gpu_memory','stage':name,'gpu_candidates':list(range(2,8)),'required_free_mib':need_mib,'pipeline_pid':os.getpid(),'time':time.time(),'note':'User authorizes GPU sharing; wait only for enough free VRAM, never stop others.'})
      if time.time()>=deadline:raise TimeoutError('Original stage/worker budget waiting VRAM')
      time.sleep(10)
   env['VSTATS_STAGE_DEADLINE']=str(min(worker_deadline,(meta['first_started'] or time.time())+4*3600))
   env['VSTATS_COMPACT_MICRO_BATCH']=str(meta['micro_batch'])
   env['VSTATS_COMPACT_AR_BATCH']=str(meta['ar_batch'])
   env['VSTATS_COMPACT_AR_MEMORY_UTILIZATION']=str(meta['ar_memory_utilization'])
   guard()
   if meta['first_started'] is None:meta['first_started']=time.time()
   meta['attempts']+=1;write(stage_meta,meta)
   cmd=[ARPY if is_ar else TRAINPY,'-B',str(Path(__file__).resolve())]+args
   with log.open('a') as f:
    f.write(json.dumps({'stage_attempt':meta['attempts'],'time':time.time(),'gpu':chosen,'micro_batch':meta['micro_batch'],'ar_batch':meta['ar_batch'],'ar_memory_utilization':meta['ar_memory_utilization'],'allocator_budget_mib':env.get('VSTATS_COMPACT_ALLOCATOR_BUDGET_MIB')})+'\n');f.flush()
    child=subprocess.Popen(cmd,cwd=P,stdout=f,stderr=subprocess.STDOUT,start_new_session=True,env=env)
    write(root/'status.json',{'state':'running','stage':name,'pid':child.pid,'pipeline_pid':os.getpid(),'gpu':chosen,'log':str(log),'command':cmd,'time':time.time(),'shared_gpu':True,'micro_batch':meta['micro_batch'],'effective_batch':8 if not is_ar else None,'ar_batch':meta['ar_batch'],'allocator_budget_mib':env.get('VSTATS_COMPACT_ALLOCATOR_BUDGET_MIB'),'physical_reserve_gib':8})
    try:code=child.wait(timeout=max(1,min(worker_deadline,meta['first_started']+4*3600)-time.time()))
    except subprocess.TimeoutExpired:
     os.killpg(child.pid,signal.SIGTERM)
     try:child.wait(timeout=30)
     except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()
     raise TimeoutError('Original stage/worker budget '+name)
   if code==0:
    if artifact:assert complete(R/artifact),'Stage exited without completed required artifact '+str(artifact)
    return
   tail=log.read_text()[-16000:].lower()
   oom=any(x in tail for x in ['cuda out of memory','cuda error: out of memory','torch.outofmemoryerror'])
   pressure='resource_headroom_low' in tail
   if not (oom or pressure):raise RuntimeError(f'Stage {name} exit {code}; see {log}')
   event={'time':time.time(),'stage':name,'failed_child_pid':child.pid,'gpu':chosen,'exit_code':code,'oom':oom,'headroom_pressure':pressure,'attempt_before':dict(meta),'source_sha256':sha(__file__),'resume':'Existing formal last.pt restores optimizer and all RNG; absent checkpoint restarts fixed init. No other process stopped.'}
   if oom:
    if is_ar:
     if meta['ar_batch']==1 and meta['ar_memory_utilization']<=.2:raise RuntimeError('AR OOM at smallest resource configuration; stop for review')
     meta['ar_batch']=max(1,meta['ar_batch']//2);meta['ar_memory_utilization']=max(.2,round(meta['ar_memory_utilization']-.05,2))
    else:
     if meta['micro_batch']==1:raise RuntimeError('Training OOM at microbatch1; stop for review')
     meta['micro_batch']=max(1,meta['micro_batch']//2)
   event['attempt_next']=dict(meta);write(stage_meta,meta)
   with (root/'resource_retries.jsonl').open('a') as f:f.write(json.dumps(event)+'\n')
  finally:
   if gpu_lock is not None:fcntl.flock(gpu_lock,fcntl.LOCK_UN);gpu_lock.close()

if __name__=="__main__":main()
