import json
from copy import deepcopy
from types import SimpleNamespace

import pytest

import scripts.evaluate_finite_final_choice as module
from scripts.final_choice_capture import capture_case

def captured():
    artifacts={'weight':'a'*64}
    def item(text,score,kind='neural'):
        return {'text':text,'score':score,'segments':[{'input_start':0,'input_end':6,'source':'imnida','text':text,'kind':kind}]}
    normal={'input':'imnida','context':'','engine':'neural','mode':'batch','raw':False,
            'version':module.RUNTIME_VERSION,'engine_ms':1.,'ranker':'hybrid',
            'model_artifacts':artifacts,'result':'입니다',
            'candidates':[item('입니다',2.),item('임니다',1.),item('imnida',None,'raw')],
            'context_used':False,'external_context_used':False,'ranking_applied':True,
            'sentence_ranking_applied':True,'warnings':[]}
    raw=dict(normal,raw=True,result='imnida',ranker=None,model_artifacts={},
             candidates=[item('imnida',0.,'raw')])
    ranker=SimpleNamespace(can_score=lambda text:True)
    converter=SimpleNamespace(sentence_ranker=ranker,
        convert=lambda source,context,raw=False:deepcopy(raw_response if raw else normal))
    raw_response=raw
    selected={'id':'id','group_id':'g','origin':'natural_eval_v3',
              'input':'imnida','context':'','expected':['입니다']}
    row=capture_case(converter,selected,expected_runtime_version=module.RUNTIME_VERSION,
                     expected_model_artifacts=artifacts)
    return row,selected,{'accepted_weight_hashes':artifacts}

def test_real_capture_validates_without_rewriting_original_origin():
    row,selected,runtime=captured();before=deepcopy((row,selected))
    module.validate_fresh_row(row,selected,runtime)
    assert (row,selected)==before

@pytest.mark.parametrize('side',['row','selection','both'])
def test_origin_mismatch_is_not_normalized_away(side):
    row,selected,runtime=captured()
    if side in ('row','both'):row['origin']='natural_eval_v2'
    if side in ('selection','both'):selected['origin']='natural_eval_v2'
    with pytest.raises(ValueError,match='origin'):
        module.validate_fresh_row(row,selected,runtime)

def test_selector_gate_precedes_fresh_source_and_model_access(tmp_path,monkeypatch):
    monkeypatch.setattr(module,'verify_hashes',lambda _:None)
    monkeypatch.setattr(module,'read_runtime_evidence',lambda *args:(b'{}',{}))
    monkeypatch.setattr(module,'validate_evaluation_partition',lambda *args:(b'{}',{}))
    def reject(*args):raise ValueError('selector not frozen')
    def forbidden(*args,**kwargs):raise AssertionError('fresh data/model accessed early')
    monkeypatch.setattr(module,'load_selector',reject)
    monkeypatch.setattr(module,'read_sources',forbidden)
    monkeypatch.setattr(module,'NeuralConverter',forbidden)
    args=SimpleNamespace(output=tmp_path/'out',selector_report=tmp_path/'selector.json',
                         selector_report_sha256='a'*64,selector_checkpoint_sha256='b'*64)
    with pytest.raises(ValueError,match='not frozen'):module.run(args)
    assert not args.output.exists()

def test_source_adapter_uses_pinned_metadata_selection(tmp_path,monkeypatch):
    monkeypatch.setattr(module,'ROOT',tmp_path)
    paths={'RAW':'raw.jsonl','V1':'v1.json','V2':'v2.json'}
    for key,value in paths.items():monkeypatch.setattr(module,key,value)
    rows=[{'id':'id','group_id':'g','split':'train','text':'입니다','left_context':'앞 문맥'}]
    content={'raw.jsonl':json.dumps(rows[0],ensure_ascii=False).encode()+b'\n',
             'v1.json':b'{"natural": {}}','v2.json':b'{}'}
    for name,raw in content.items():(tmp_path/name).write_bytes(raw)
    monkeypatch.setattr(module,'PINS',{name:module.digest(raw) for name,raw in content.items()})
    partition={'eval_ids':['id'],'eval_group_ids':['g'],'remaining_ids':[], 'remaining_group_ids':[]}
    def select(metadata,natural,previous):
        assert metadata==[{'id':'id','group_id':'g'}]
        return partition
    monkeypatch.setattr(module,'select_ids',select)
    sources,_=module.read_sources(partition)
    assert sources==[{'id':'id','group_id':'g','origin':'natural_eval_v3',
                      'input':'imnida','context':'앞 문맥','expected':['입니다']}]

def test_metadata_pin_precedes_fresh_source_access(tmp_path,monkeypatch):
    monkeypatch.setattr(module,'verify_hashes',lambda _:None)
    monkeypatch.setattr(module,'read_runtime_evidence',lambda *args:(b'{}',{}))
    monkeypatch.setattr(module,'validate_evaluation_partition',lambda *args:(b'{}',{}))
    monkeypatch.setattr(module,'load_selector',lambda *args:{'metadata_raw':b'changed metadata'})
    def forbidden(*args):raise AssertionError('fresh sources accessed with changed metadata')
    monkeypatch.setattr(module,'read_sources',forbidden)
    args=SimpleNamespace(output=tmp_path/'out',selector_report=tmp_path/'selector.json',
                         selector_report_sha256='a'*64,selector_checkpoint_sha256='b'*64,
                         selector_metadata_sha256='c'*64)
    with pytest.raises(ValueError,match='metadata hash'):module.run(args)
    assert not args.output.exists()

def test_hash_merge_rejects_conflicting_provenance():
    with pytest.raises(ValueError,match='conflict'):
        module.merge({'file':'a'*64},{'file':'b'*64})

def save_capture(directory,rows,sources):
    cases_raw=b''.join((json.dumps(r,ensure_ascii=False)+'\n').encode() for r in rows)
    selection_raw=b''.join((json.dumps(r,ensure_ascii=False)+'\n').encode() for r in sources)
    (directory/'parent-capture.jsonl').write_bytes(cases_raw)
    (directory/'selection.jsonl').write_bytes(selection_raw)
    manifest={'rows':len(sources),'cases_sha256':module.digest(cases_raw),
              'selection_sha256':module.digest(selection_raw)}
    raw=json.dumps(manifest).encode();(directory/'capture-manifest.json').write_bytes(raw)
    return module.digest(raw)

def test_file_capture_is_reloaded_as_new_validated_objects(tmp_path,monkeypatch):
    monkeypatch.setattr(module,'ROOT',tmp_path)
    row,selected,runtime=captured()
    digest=save_capture(tmp_path,[row],[selected])
    rows,hashes=module.load_fresh_parent_capture(tmp_path,[selected],digest,runtime,[row])
    assert rows==[row] and rows[0] is not row
    assert set(hashes)=={'parent-capture.jsonl','selection.jsonl','capture-manifest.json'}

@pytest.mark.parametrize('mutation',['bytes','json','changed_row','selection','order'])
def test_corrupt_or_changed_disk_capture_fails(tmp_path,monkeypatch,mutation):
    monkeypatch.setattr(module,'ROOT',tmp_path)
    row,selected,runtime=captured()
    memory=[deepcopy(row)];sources=[deepcopy(selected)]
    digest=save_capture(tmp_path,memory,sources)
    if mutation=='bytes':
        with (tmp_path/'parent-capture.jsonl').open('ab') as stream:stream.write(b' ')
    elif mutation=='json':
        data=b'not JSON';(tmp_path/'parent-capture.jsonl').write_bytes(data)
        manifest={'rows':1,'cases_sha256':module.digest(data),
                  'selection_sha256':module.digest((tmp_path/'selection.jsonl').read_bytes())}
        raw=json.dumps(manifest).encode();(tmp_path/'capture-manifest.json').write_bytes(raw);digest=module.digest(raw)
    elif mutation=='changed_row':
        row['engine_ms']=123.
        digest=save_capture(tmp_path,[row],sources)
    elif mutation=='selection':
        selected['context']='changed'
        digest=save_capture(tmp_path,[row],[selected])
    else:
        second=deepcopy(row);second['id']='second'
        other=deepcopy(selected);other['id']='second'
        memory.append(second);sources.append(other)
        digest=save_capture(tmp_path,list(reversed(memory)),sources)
    with pytest.raises(ValueError):
        module.load_fresh_parent_capture(tmp_path,sources,digest,runtime,memory)
