import hashlib
import json
from pathlib import Path
import random

import pytest
import torch
from torch import nn

import scripts.sentence_transducer_recovery as recovery


class Tiny(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(nn.Linear(4, 8), nn.Dropout(.35), nn.Linear(8, 2))

    def forward(self, value):
        return self.network(value)


def state(updates=0, next_batch=0):
    return {"updates": updates, "next_epoch": 0, "next_batch": next_batch,
            "epochs_report": [], "epoch_loss_sum": 0.0, "epoch_token_sum": 0,
            "training_elapsed_s": 2.0, "peak_rss_bytes": 100}


def step(model, optimizer, value):
    model.train();optimizer.zero_grad(set_to_none=True)
    loss=model(value + random.random()).square().mean();loss.backward()
    norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
    optimizer.step()
    return float(loss.detach()),float(norm)


def setup(seed=19):
    torch.manual_seed(seed);random.seed(seed)
    model=Tiny();optimizer=torch.optim.AdamW(model.parameters(),lr=.01)
    return model,optimizer


def tensors(value):
    if torch.is_tensor(value):yield value
    elif isinstance(value,dict):
        for item in value.values():yield from tensors(item)
    elif isinstance(value,(list,tuple)):
        for item in value:yield from tensors(item)


def assert_nested_equal(left,right):
    if torch.is_tensor(left):assert torch.equal(left,right)
    elif isinstance(left,dict):
        assert left.keys()==right.keys()
        for key in left:assert_nested_equal(left[key],right[key])
    elif isinstance(left,(list,tuple)):
        assert len(left)==len(right)
        for a,b in zip(left,right):assert_nested_equal(a,b)
    else:assert left==right


def test_dropout_optimizer_resume_matches_uninterrupted_and_rolls_back_tail(tmp_path):
    values=[torch.arange(8,dtype=torch.float32).reshape(2,4)+i for i in range(6)]
    full_model,full_optimizer=setup();full=[step(full_model,full_optimizer,value) for value in values]
    model,optimizer=setup();partial=[step(model,optimizer,value) for value in values[:3]]
    log=tmp_path/'updates.jsonl'
    with log.open('w+') as stream:
        for index,result in enumerate(partial,1):
            stream.write(json.dumps({'step':index,'loss':result[0]})+'\n')
        recovery.save_checkpoint(tmp_path,model,optimizer,{'run':'same'},state(3,3),stream)
        tail_result=step(model,optimizer,values[3])
        stream.write(json.dumps({'step':4,'loss':tail_result[0],
            'training_elapsed_s':9.0,'peak_rss_bytes':250})+'\n');stream.flush()
    resumed_model,resumed_optimizer=setup(seed=999)
    restored=recovery.restore_checkpoint(tmp_path,resumed_model,resumed_optimizer,{'run':'same'})
    resumed=[step(resumed_model,resumed_optimizer,value) for value in values[3:]]
    assert resumed==full[3:]
    assert_nested_equal(resumed_model.state_dict(),full_model.state_dict())
    assert_nested_equal(resumed_optimizer.state_dict(),full_optimizer.state_dict())
    assert restored['updates']==3 and restored['next_batch']==3
    assert restored['discarded_tail_updates']==1
    assert restored['training_elapsed_s']==9.0 and restored['peak_rss_bytes']==250
    assert Path(restored['discarded_tail_archive']).read_text().count('\n')==1
    assert len(log.read_text().splitlines())==3

    with log.open('a+') as stream:
        recovery.save_checkpoint(tmp_path,resumed_model,resumed_optimizer,
                                 {'run':'same'},restored,stream)
        stream.write(json.dumps({'step':5,'training_elapsed_s':11.0,
                                 'peak_rss_bytes':275})+'\n')
    twice_model,twice_optimizer=setup(seed=1000)
    restored_twice=recovery.restore_checkpoint(tmp_path,twice_model,twice_optimizer,
                                                {'run':'same'})
    assert restored_twice['discarded_tail_updates']==2
    assert restored_twice['training_elapsed_s']==11.0
    assert restored_twice['peak_rss_bytes']==275


def test_restore_rejects_identity_and_corrupt_or_short_prefix(tmp_path):
    model,optimizer=setup();log=tmp_path/'updates.jsonl'
    with log.open('w+') as stream:
        stream.write('{"step":1}\n')
        recovery.save_checkpoint(tmp_path,model,optimizer,{'run':'one'},state(1,1),stream)
    other_model,other_optimizer=setup(seed=20)
    with pytest.raises(ValueError,match='identity'):
        recovery.restore_checkpoint(tmp_path,other_model,other_optimizer,{'run':'two'})
    original=log.read_bytes();log.write_bytes(b'X'+original[1:])
    with pytest.raises(ValueError,match='prefix differs'):
        recovery.restore_checkpoint(tmp_path,other_model,other_optimizer,{'run':'one'})
    log.write_bytes(original[:-1])
    with pytest.raises(ValueError,match='shorter'):
        recovery.restore_checkpoint(tmp_path,other_model,other_optimizer,{'run':'one'})


def test_incompatible_model_does_not_archive_or_truncate_tail(tmp_path):
    model,optimizer=setup();log=tmp_path/'updates.jsonl'
    with log.open('w+') as stream:
        stream.write('{"step":1}\n')
        recovery.save_checkpoint(tmp_path,model,optimizer,{},state(1,1),stream)
        stream.write('{"step":2}\n');stream.flush()
    original=log.read_bytes()
    incompatible=nn.Linear(3,1)
    incompatible_optimizer=torch.optim.AdamW(incompatible.parameters(),lr=.01)
    with pytest.raises(RuntimeError):
        recovery.restore_checkpoint(tmp_path,incompatible,incompatible_optimizer,{})
    assert log.read_bytes()==original
    assert not list(tmp_path.glob('*.uncommitted-*'))


def test_nonfinite_model_and_optimizer_states_are_rejected(tmp_path):
    model,optimizer=setup();step(model,optimizer,torch.ones(2,4));log=tmp_path/'updates.jsonl'
    with log.open('w+') as stream:
        with torch.no_grad():next(model.parameters()).view(-1)[0]=float('nan')
        with pytest.raises(ValueError,match='model state'):
            recovery.save_checkpoint(tmp_path,model,optimizer,{},state(),stream)
    model,optimizer=setup();step(model,optimizer,torch.ones(2,4))
    next(iter(optimizer.state.values()))['exp_avg'].view(-1)[0]=float('inf')
    with log.open('w+') as stream:
        with pytest.raises(ValueError,match='optimizer state'):
            recovery.save_checkpoint(tmp_path,model,optimizer,{},state(),stream)


def test_failed_temporary_write_preserves_prior_checkpoint(tmp_path,monkeypatch):
    model,optimizer=setup();log=tmp_path/'updates.jsonl'
    with log.open('w+') as stream:
        metadata=recovery.save_checkpoint(tmp_path,model,optimizer,{},state(),stream)
        before=(tmp_path/'recovery.pt').read_bytes()
        def fail(*args,**kwargs):
            args[1].write(b'partial');raise OSError('synthetic failure')
        monkeypatch.setattr(recovery.torch,'save',fail)
        with pytest.raises(OSError):recovery.save_checkpoint(tmp_path,model,optimizer,{},state(),stream)
    assert (tmp_path/'recovery.pt').read_bytes()==before
    assert hashlib.sha256(before).hexdigest()==metadata['checkpoint_sha256']
    assert not (tmp_path/'.recovery.pt.tmp').exists()


@pytest.mark.parametrize('key,value',[('updates',-1),('next_epoch',True),('epoch_loss_sum',float('nan')),
                                      ('training_elapsed_s',-1.),('peak_rss_bytes',-1)])
def test_invalid_recovery_counters_are_rejected(tmp_path,key,value):
    model,optimizer=setup();log=tmp_path/'updates.jsonl';broken=state();broken[key]=value
    with log.open('w+') as stream:
        with pytest.raises(ValueError):recovery.save_checkpoint(tmp_path,model,optimizer,{},broken,stream)
