"""Program-verifiable tasks and split-disjoint prompt grammar; no EEG or teacher access."""

if __name__ == "__main__":
    raise SystemExit("Reference snapshot only. Use tools/workbench.py for inspection; configure a fresh run before training.")

from pathlib import Path
import copy,hashlib,itertools,json,random,re
import oracle_robust28_1016 as old
base=old.base;FIELDS=old.FIELDS;VALUES=old.VALUES;SEED=1014
R=old.P/'experiments/mainline_vrl/runs/oracle_curriculum28_20261016_s1014'
FAMILIES=['read','partial_report','nested_report','rule','compare','count','select','policy','report']
FRAMES={
 'train':['Complete this observation request.','Fill the requested evidence form.','Use the attached observations to resolve this query.','Prepare a structured response for the workflow.','Answer from the evidence rather than assumptions.','Process the record query below.','Return the requested measurements and calculations.','Build a compact factual response.'],
 'val':['Resolve this evidence enquiry in the specified structure.','Translate these observations into the requested result.','Work through the attached measurement request.','Produce an evidence-grounded response to the following query.'],
 'test':['Reconcile the supplied record evidence with this output request.','Deliver the requested structured interpretation of the observations.','Read the evidence and complete the requested calculation.','Construct the specified response from the available observations.']}
SYSTEMS={
 'train':['You answer structured observation requests.','You assist a record-processing workflow.','Follow the user schema and use only supplied evidence.',''],
 'val':['You are preparing a factual observation register.','You handle evidence queries for a data export tool.'],
 'test':['You help populate a machine-readable observation archive.','You resolve structured queries over supplied records.']}
NOTES={
 'train':['The export filename is an administrative identifier.','The folder address carries no measurement.','A scheduling entry is unrelated to the observations.','This filing instruction adds no signal evidence.','The operator may save the requested result.','A display preference does not change the recorded facts.'],
 'val':['The coordinator is organizing document retrieval.','The archive date describes filing rather than physiology.','A form revision does not add measured evidence.','The destination account only determines routing.'],
 'test':['The batch title is used for document lookup.','The storage policy does not describe any waveform.','An index entry identifies a record but changes no measurement.','The page owner manages files rather than signal content.']}
VERBS={
 'train':['Retrieve','Read','Extract','Decode'],
 'val':['Identify','Find'],
 'test':['Recover','Report']}
TAILS={
 'train':['Return JSON only, without a Markdown fence.','Emit exactly the requested JSON object and no surrounding prose.','Your response must be one JSON object with no extra keys.','Write only the structured object requested below.'],
 'val':['Respond with the sole requested JSON object, without introductory text.','Serialize the specified structure directly; do not add a code block.'],
 'test':['Deliver the requested object as plain JSON, with no other material.','Use the specified keys and output only valid JSON.']}
def dumps(x):return json.dumps(x,separators=(',',':'),ensure_ascii=False)
def facts(r):return old.actual_values(r)
def value(r,f):return r['label'] if f=='label' else facts(r)[f]
def relation(a,b):
 rank=dict(zip(VALUES,range(3)))
 return 'equal' if a==b else 'lower' if rank[a]<rank[b] else 'higher'
def rule_expr(pattern,fs,vv,flag):
 lit=['('+f+' is '+v+')' for f,v in zip(fs,vv)]
 if pattern=='all':return ' AND '.join(lit)
 if pattern=='any':return ' OR '.join(lit)
 if pattern=='text_train':return '(enabled AND ('+lit[0]+' OR '+lit[1]+')) OR '+lit[2]
 if pattern=='val_holdout':return 'enabled AND ('+lit[0]+' OR ('+lit[1]+' AND '+lit[2]+'))'
 if pattern=='test_holdout':return '(enabled OR '+lit[0]+') AND ('+lit[1]+' OR '+lit[2]+')'
 raise ValueError(pattern)
def rule_truth(pattern,m,flag):
 if pattern=='all':return all(m)
 if pattern=='any':return any(m)
 if pattern=='text_train':return (flag and (m[0] or m[1])) or m[2]
 if pattern=='val_holdout':return flag and (m[0] or (m[1] and m[2]))
 if pattern=='test_holdout':return (flag or m[0]) and (m[1] or m[2])
 raise ValueError(pattern)
def specification(rs,family,rng,logic=None):
 f=rng.choice(['label']+FIELDS) if family=='read' else rng.choice(FIELDS)
 spec={'family':family,'field':f,'desired':rng.choice(VALUES),'flag':bool(rng.randrange(2)),'threshold':rng.randrange(1,len(rs)+1)}
 if family in ['partial_report','nested_report']:
  spec['fields']=rng.sample(FIELDS,3 if family=='partial_report' else 6)
 if family=='rule':
  spec['fields']=rng.sample(FIELDS,3);spec['pattern']=logic or rng.choice(['all','any','text_train'])
  possible={rule_truth(spec['pattern'],m,spec['flag']) for m in itertools.product([False,True],repeat=3)}
  desired_truth=rng.choice(sorted(possible))
  while True:
   vv=[rng.choice(VALUES) for _ in range(3)];m=[value(rs[0],f)==v for f,v in zip(spec['fields'],vv)]
   if rule_truth(spec['pattern'],m,spec['flag'])==desired_truth:break
  spec['literals']=vv
 return spec
def truth(rs,s):
 fam=s['family'];f=s['field'];obs=[];checks=[]
 def observe(i,field):
  v=value(rs[i],field);obs.append({'record':chr(65+i),'field':field,'value':v});return v
 if fam=='read':result={f:observe(0,f)}
 elif fam in ['partial_report','nested_report']:
  lab=observe(0,'label');ff={f:observe(0,f) for f in s['fields']}
  result={'label':lab,'statistics':ff} if fam=='partial_report' else {'record':{'class':lab,'measurements':[{'name':f,'level':ff[f]} for f in s['fields']]}}
 elif fam=='rule':
  m=[observe(0,f)==v for f,v in zip(s['fields'],s['literals'])]
  ok=rule_truth(s['pattern'],m,s['flag'])
  checks=[{'name':'matches','value':m},{'name':'rule_result','value':ok}]
  result={'decision':'review' if ok else 'routine'}
 else:
  levels=[observe(i,f) for i in range(len(rs))];cnt=levels.count(s['desired']);rel=relation(levels[0],levels[-1])
  selected=[chr(65+i) for i,v in enumerate(levels) if v==s['desired']]
  act='review' if s['flag'] and cnt>=s['threshold'] else 'routine'
  if fam=='compare':checks=[{'name':'first_vs_last','value':rel}];result={'levels':levels,'relation':rel}
  elif fam=='count':checks=[{'name':'matching_count','value':cnt}];result={'levels':levels,'count':cnt}
  elif fam=='select':checks=[{'name':'matching_records','value':selected}];result={'levels':levels,'records':selected}
  elif fam=='policy':checks=[{'name':'matching_count','value':cnt},{'name':'enabled','value':s['flag']}];result={'levels':levels,'action':act}
  elif fam=='report':
   checks=[{'name':'first_vs_last','value':rel},{'name':'matching_count','value':cnt},{'name':'enabled','value':s['flag']}]
   result={'levels':levels,'first_vs_last':rel,'selected_count':cnt,'action':act}
  else:raise ValueError(fam)
 return {'trace':{'observations':obs,'checks':checks},'result':result}
def schema(x,path=()):
 if isinstance(x,dict):return {k:schema(v,path+(k,)) for k,v in x.items()}
 if isinstance(x,list):
  if path and path[-1]=='checks':return [{'name':v['name'],'value':(['matching record letters, possibly empty'] if v['name']=='matching_records' else 'lower|equal|higher' if v['name']=='first_vs_last' else schema(v['value'],path+('value',)))} for v in x]
  if path and path[-1]=='measurements':return [{'name':v['name'],'level':'low|mid|high'} for v in x]
  if path and path[-1]=='observations':return [{'record':v['record'],'field':v['field'],'value':'CLASS' if v['field']=='label' else 'low|mid|high'} for v in x]
  if path and path[-1] in ['records','matching_records']:return ['matching record letters, possibly empty']
  return [schema(v,path+('item',)) for v in x]
 if type(x)==bool:return 'true|false'
 if type(x)==int:return 'integer'
 if path and path[-1] in ['relation','first_vs_last']:return 'lower|equal|higher'
 if x in VALUES:return 'low|mid|high'
 if x in ['review','routine']:return 'review|routine'
 return 'CLASS'
def prompt(rs,s,rng,split,suite):
 fam=s['family'];names=[chr(65+i) for i in range(len(rs))];last=names[-1];f=s['field']
 verb=rng.choice(VERBS[split])
 if fam=='read':body=verb+' the '+f+' observation for record A.'
 elif fam in ['partial_report','nested_report']:
  body=verb+' the class and these measurements for record A: '+', '.join(s['fields'])+'. Preserve the listed order where an array is requested.'
 elif fam=='rule':
  body=verb+' the three measurements for record A. External flag enabled='+str(s['flag']).lower()+'. Evaluate ('+rule_expr(s['pattern'],s['fields'],s['literals'],s['flag'])+'). Choose review when true, otherwise routine.'
 else:
  body=verb+' '+f+' for records '+', '.join(names)+' in that order. The ordering is low < mid < high. '
  if fam=='compare':body+='Compare A relative to '+last+'; relation is lower, equal or higher.'
  elif fam=='count':body+='Count records at level '+s['desired']+'.'
  elif fam=='select':body+='Select records at level '+s['desired']+' as record letters in input order, or [] when none match.'
  elif fam=='policy':body+='External flag enabled='+str(s['flag']).lower()+'. Choose review exactly if enabled AND the count of '+s['desired']+' records is at least '+str(s['threshold'])+'; otherwise routine.'
  elif fam=='report':body+='Prepare a factual summary with the levels, A relative to '+last+', the count of '+s['desired']+' records, and action. External flag enabled='+str(s['flag']).lower()+'. Action is review exactly if enabled AND that count is at least '+str(s['threshold'])+'; otherwise routine.'
 n=rng.choice([0,4,8,16]) if suite!='long_context' else (rng.choice([24,32]) if split=='val' else rng.choice([40,48]))
 ctx='\n'.join('Filing note '+str(i)+': '+rng.choice(NOTES[split]) for i in range(n))
 text=rng.choice(FRAMES[split])+'\n'+(ctx+'\n' if ctx else '')+body+'\nList the observations and operation checks in trace, then fill result. Use precisely this schema (placeholders are not observed values): '+dumps(schema(truth(rs,s)))+'\n'+rng.choice(TAILS[split])
 system=rng.choice(SYSTEMS[split])
 history=[{'role':'user','content':'Prepare to process the next observation query; no measurements are provided in this message.'},{'role':'assistant','content':'Ready for the observation query.'}] if suite=='role_context' else []
 return text,system,history,rng.choice(['evidence_first','question_first']),n
def make(rs,s,rng,split,suite,ident,wording=None):
 rs=copy.deepcopy(rs);wording=wording or split
 text,system,history,layout,n=prompt(rs,s,rng,wording,suite)
 return {'id':ident,'records':rs,'spec':s,'expected':truth(rs,s),'question':text,'system':system,'history':history,'layout':layout,'context_lines':n,'scope':suite,'dataset':rs[0]['dataset'],'kind':'process','split':split}
def draw_train(round_,i,labels):
 rng=random.Random(SEED+round_*1000000+i*97);fam=FAMILIES[i%len(FAMILIES)];n=rng.choice([2,4]) if fam in FAMILIES[4:] else 1
 rs=[base.synthetic_record(rng,labels,'train',round_*10000000+i*4+j) for j in range(n)]
 return make(rs,specification(rs,fam,rng),rng,'train','teacher_train',f'train:r{round_}:{i}')
def canonical(e):
 return {'id':e['id'],'records':e['records'],'q':e['q'],'kind':e['kind'],'scope':'original_base','dataset':e['dataset'],'split':e['records'][0]['split']}
def expected(e):return e['expected'] if e['kind']=='process' else json.loads(old.target(e))
def target(e):return dumps(expected(e))
def render(tok,emb,e,prefix=None,teacher=False,reference=False):
 if e['kind']!='process':
  x=old.render(tok,emb,e,prefix,teacher)
  if not reference:return x
  # Build reference context before the assistant boundary, never student input.
  bef,aft=base.wrapper(tok);body='Evidence:\n'+dumps([{'label':r['label'],'statistics':facts(r)} for r in e['records']])+'\n'+e['q']['question']+'\nVerified reference solution for training guidance:\n'+target(e)+'\nProduce the requested response, using the reference to verify every fact.'
  ids=base.token_ids(tok,bef+body+aft);return emb(__import__('torch').tensor(ids,device=emb.weight.device)).detach()
 messages=[]
 if e.get('system'):messages.append({'role':'system','content':e['system']})
 messages+=e.get('history',[])
 evidence='Observations for independent records; order does not imply elapsed time.\n'
 for i,r in enumerate(e['records']):
  evidence+='Record '+chr(65+i)+':\n'+(dumps({'label':r['label'],'statistics':facts(r)}) if teacher else '__ORACLE_'+str(i)+'__')+'\n'
 body=evidence+'\n'+e['question'] if e['layout']=='evidence_first' else e['question']+'\n'+evidence
 if reference:body+='\nVerified reference solution for training guidance:\n'+target(e)+'\nFollow the requested schema and verify against this reference.'
 messages.append({'role':'user','content':body})
 text=tok.apply_chat_template(messages,tokenize=False,add_generation_prompt=True,enable_thinking=False)
 import torch
 parts=[]
 for piece in re.split(r'(__ORACLE_\d+__)',text):
  if re.fullmatch(r'__ORACLE_\d+__',piece):
   i=int(piece[len('__ORACLE_'):-2]);parts.append(prefix[i].to(device=emb.weight.device,dtype=emb.weight.dtype))
  elif piece:parts.append(emb(torch.tensor(base.token_ids(tok,piece),device=emb.weight.device,dtype=torch.long)).detach())
 return torch.cat(parts)
def normalize_teacher(text):
 s=text.strip()
 if s.startswith('~~~') or s.startswith('~~~json'):return s,False
 if s.startswith(chr(96)*3):
  s=re.sub(r'^'+chr(96)*3+r'(?:json)?\s*','',s);s=re.sub(r'\s*'+chr(96)*3+r'$','',s)
  return s.strip(),True
 return s,False
def same(a,b):
 return type(a)==type(b) and (set(a)==set(b) and all(same(a[k],b[k]) for k in a) if isinstance(a,dict) else len(a)==len(b) and all(same(x,y) for x,y in zip(a,b)) if isinstance(a,list) else a==b)
def shape(a,b,path=()):
 if isinstance(b,dict):
  if set(b)=={'name','value'} and b.get('name')=='matching_records':
   return isinstance(a,dict) and set(a)==set(b) and isinstance(a.get('value'),list) and all(isinstance(v,str) and re.fullmatch('[A-D]',v) for v in a['value'])
  return isinstance(a,dict) and set(a)==set(b) and all(shape(a[k],v,path+(k,)) for k,v in b.items())
 if isinstance(b,list):
  if path and path[-1]=='records':return isinstance(a,list) and all(isinstance(v,str) and re.fullmatch('[A-D]',v) for v in a)
  return isinstance(a,list) and len(a)==len(b) and all(shape(x,y,path+(str(i),)) for i,(x,y) in enumerate(zip(a,b)))
 if type(b)==bool:return type(a)==bool
 if type(b)==int:return type(a)==int and a>=0
 if b in VALUES:return a in VALUES
 if b in ['review','routine']:return a in ['review','routine']
 if b in ['lower','equal','higher']:return a in ['lower','equal','higher']
 return isinstance(a,str)
def parse(text,e):
 if e['kind']!='process':
  p,v,f,d,x=old.parse(text,e)
  return {'prediction':p,'valid':v,'fact_score':f,'decision_score':d if d is None or v else 0.,'correct':bool(x),'trace_correct':None,'fact_items':[]}
 try:p=json.loads(text.strip())
 except (ValueError,TypeError):p=None
 gold=e['expected'];v=shape(p,gold);obs=(p or {}).get('trace',{}).get('observations',[]) if isinstance(p,dict) and isinstance(p.get('trace'),dict) else []
 fi=[]
 for j,t in enumerate(gold['trace']['observations']):
  a=obs[j] if j<len(obs) and isinstance(obs[j],dict) else {}
  pred=a.get('value') if a.get('field')==t['field'] and a.get('record')==t['record'] else None
  fi.append({'field':t['field'],'gold':t['value'],'prediction':pred,'correct':type(pred)==type(t['value']) and pred==t['value']})
 v=v and len(obs)==len(gold['trace']['observations']) and all(isinstance(a,dict) and a.get('record')==t['record'] and a.get('field')==t['field'] for a,t in zip(obs,gold['trace']['observations']))
 if isinstance(p,dict) and isinstance(p.get('trace'),dict):
  cc=p['trace'].get('checks',[]);gg=gold['trace']['checks']
  v=v and isinstance(cc,list) and len(cc)==len(gg) and all(isinstance(a,dict) and a.get('name')==b['name'] for a,b in zip(cc,gg))
 if e['spec']['family']=='nested_report' and isinstance(p,dict):
  got=p.get('result',{}).get('record',{}).get('measurements',[]) if isinstance(p.get('result'),dict) and isinstance(p['result'].get('record'),dict) else []
  v=v and isinstance(got,list) and [x.get('name') for x in got if isinstance(x,dict)]==e['spec']['fields']
 fam=e['spec']['family'];dec=None
 if fam=='select' and isinstance(p,dict):
  names={chr(65+i) for i in range(len(e['records']))}
  rr=p.get('result',{}).get('records',[]) if isinstance(p.get('result'),dict) else []
  cc=p.get('trace',{}).get('checks',[]) if isinstance(p.get('trace'),dict) else []
  vv=cc[0].get('value',[]) if cc and isinstance(cc[0],dict) else []
  v=v and isinstance(rr,list) and isinstance(vv,list) and all(x in names for x in rr+vv)
 if fam in ['rule','compare','count','select','policy','report']:
  aa=p.get('result',{}) if isinstance(p,dict) else {}
  keys={'rule':['decision'],'compare':['relation'],'count':['count'],'select':['records'],'policy':['action'],'report':['first_vs_last','selected_count','action']}[fam]
  dec=float(v and all(k in aa and same(aa[k],gold['result'][k]) for k in keys))
 return {'prediction':p,'valid':bool(v),'fact_score':sum(x['correct'] for x in fi)/len(fi),'decision_score':dec,'correct':bool(v and same(p,gold)),'trace_correct':bool(isinstance(p,dict) and same(p.get('trace'),gold['trace'])),'fact_items':fi}
