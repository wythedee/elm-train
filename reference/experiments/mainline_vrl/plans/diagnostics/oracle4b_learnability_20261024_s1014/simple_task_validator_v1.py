#!/usr/bin/env python3
"""Validate externally generated templates and calculate simple ordinal answers.
This file does not author training questions or launch inference.
"""

if __name__ == "__main__":
    raise SystemExit("Reference snapshot only. Use tools/workbench.py for inspection; configure a fresh run before training.")

import argparse,collections,hashlib,itertools,json,re,string
from pathlib import Path
O=Path(__file__).resolve().parent
CONTRACT=json.loads((O/'template_contract.json').read_text())
OPS=CONTRACT['operations'];LEVELS=CONTRACT['ordinal_levels']
def gold(operation,levels):
    if operation not in OPS:raise ValueError('Unknown operation')
    if len(levels)!=OPS[operation]['records']:raise ValueError('Wrong record count')
    ranks=[LEVELS.index(x) for x in levels]
    if operation=='read_level':answer=levels[0]
    elif operation=='compare_level':answer='equal' if ranks[0]==ranks[1] else 'lower' if ranks[0]<ranks[1] else 'higher'
    elif ranks[0]==ranks[1]:answer='tie'
    else:
        index=ranks.index(max(ranks) if operation=='choose_higher' else min(ranks))
        answer=['A','B'][index]
    return {'answer':answer}
def score(generation,operation,levels):
    expected=gold(operation,levels)
    try:pred=json.loads(generation)
    except (ValueError,TypeError):return {'valid':False,'correct':False,'expected':expected}
    valid=isinstance(pred,dict) and set(pred)=={'answer'} and isinstance(pred['answer'],str) and pred['answer'] in OPS[operation]['result_values']
    return {'valid':valid,'correct':bool(valid and pred==expected),'expected':expected}
def validate(row):
    assert set(row)==set(CONTRACT['template_row_keys']),'Unexpected/missing template fields'
    op=row['operation'];assert op in OPS
    assert re.fullmatch(r'[a-zA-Z0-9_-]{1,80}',row['template_id'])
    text=row['wording'];assert isinstance(text,str) and 15<=len(text)<=500
    placeholders=set()
    for literal,key,spec,conv in string.Formatter().parse(text):
        if key is not None:
            assert key in ['field','record_a','record_b'] and not spec and conv is None,'Unsafe placeholder'
            placeholders.add(key)
    assert placeholders==({'field','record_a'} if op=='read_level' else {'field','record_a','record_b'})
    assert set(row['answer_options'])==set(OPS[op]['result_values'])
    assert row['intended_direction']==op
    assert row['tie_semantics']==('not_applicable' if op=='read_level' else 'same_ordinal_bin')
    assert not re.search(r'reference solution|ground.?truth|correct answer is|ignore.*instructions',text,re.I)
    return row
def normalize(text):return re.sub(r'[^a-z0-9{}]+',' ',text.lower()).strip()
def fixture():
    cases=[]
    for op,meta in OPS.items():
        for levels in itertools.product(LEVELS,repeat=meta['records']):
            target=gold(op,levels);assert score(json.dumps(target),op,levels)['correct']
            assert not score(json.dumps({**target,'extra':'wrong'}),op,levels)['valid']
            for other in meta['result_values']:
                s=score(json.dumps({'answer':other}),op,levels)
                assert s['correct']==(other==target['answer'])
            if len(levels)==2:
                flipped=gold(op,tuple(reversed(levels)))['answer']
                expected={'lower':'higher','higher':'lower','equal':'equal'}[target['answer']] if op=='compare_level' else {'A':'B','B':'A','tie':'tie'}[target['answer']]
                assert flipped==expected
            cases.append({'operation':op,'levels':levels,'target':target})
    # Deliberately non-natural fixtures never enter any training dataset.
    row={'template_id':'FIXTURE_ONLY','operation':'compare_level','wording':'{record_a} {field} {record_b} compare_level fixture only','answer_options':['lower','equal','higher'],'intended_direction':'compare_level','tie_semantics':'same_ordinal_bin'}
    validate(row)
    for replacement in ['{field.__class__}', '{field!r}', '{field:30}', '{answer}']:
        bad=dict(row,wording=row['wording'].replace('{field}',replacement))
        try:validate(bad)
        except (AssertionError,ValueError):pass
        else:raise AssertionError('Unsafe template accepted')
    result={'status':'cpu_fixture_passed','truth_cases':len(cases),'all_target_parser_checks':True,'swap_direction_and_ties_checked':True,
            'unsafe_placeholders_rejected':4,'training_questions_authored':0,'generator_called':False,'not_a_4b_capability_measurement':True,
            'contract_sha256':hashlib.sha256((O/'template_contract.json').read_bytes()).hexdigest(),
            'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    (O/'simple_task_fixture.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result))
def validate_file(path):
    obj=json.loads(Path(path).read_text())
    # Authorship provenance is mandatory; merely labeling a file 27B is not sufficient.
    assert obj.get('model_identity') and obj.get('raw_generation_file') and obj.get('request_file')
    raw=Path(obj['raw_generation_file']);request=Path(obj['request_file'])
    assert raw.is_file() and request.is_file()
    assert hashlib.sha256(raw.read_bytes()).hexdigest()==obj['raw_generation_sha256']
    assert hashlib.sha256(request.read_bytes()).hexdigest()==obj['request_sha256']
    rows=[validate(x) for x in obj['templates']]
    assert len({x['template_id'] for x in rows})==len(rows)
    assert len({normalize(x['wording']) for x in rows})==len(rows),'Duplicate wording'
    counts=collections.Counter(x['operation'] for x in rows)
    assert all(counts[op]>=24 for op in OPS),counts
    print(json.dumps({'status':'structure_only_passed','counts':counts,'semantic_screen_with_4b_pending':True}))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('command',choices=['fixture','validate']);p.add_argument('--input');a=p.parse_args()
    if a.command=='fixture':fixture()
    else:validate_file(a.input)
