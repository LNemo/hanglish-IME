import json

import pytest
import torch

from hanglish.models.sentence_transducer import SentenceTransducerConfig
from scripts import train_sentence_transducer as trainer


@pytest.mark.parametrize('interrupt_before',[3,4])
def test_training_cursor_resume_matches_uninterrupted_weights_and_losses(tmp_path,monkeypatch,interrupt_before):
    torch.set_num_threads(2)
    rows=[dict(id=str(i),input=f'jeoneun haksaengimnida {i}',expected=['저는 학생입니다']) for i in range(33)]
    config=SentenceTransducerConfig(width=16,heads=2,encoder_layers=1,decoder_layers=1,
                                   feedforward=32,dropout=.2,max_tokens=64)
    qualification=tmp_path/'qualification.json';qualification.write_text('{}')
    monkeypatch.setattr(trainer,'qualify',lambda *args:{})
    monkeypatch.setattr(trainer,'estimate_training_seconds',lambda *args:1.)
    monkeypatch.setattr(trainer,'CHECKPOINT_INTERVAL',2)
    def empty_evaluation(model, output):
        (output/'dev-predictions.jsonl').write_text('')
        return dict(rows=0,correct=0,complete=0,forward_equal=0,elapsed_s=0.)
    monkeypatch.setattr(trainer,'evaluate',empty_evaluation)
    baseline=tmp_path/'baseline';baseline.mkdir()
    interrupted=tmp_path/'interrupted';interrupted.mkdir()
    torch.manual_seed(91)
    trainer.train(baseline,rows,{},config,qualification,2)
    original=trainer.optimizer_step
    calls=0
    def interrupt(*args,**kwargs):
        nonlocal calls
        calls+=1
        if calls==interrupt_before:raise InterruptedError('simulated process boundary')
        return original(*args,**kwargs)
    monkeypatch.setattr(trainer,'optimizer_step',interrupt)
    torch.manual_seed(91)
    with pytest.raises(InterruptedError):trainer.train(interrupted,rows,{},config,qualification,2)
    monkeypatch.setattr(trainer,'optimizer_step',original)
    torch.manual_seed(888)  # Construction must not replace the checkpoint's RNG.
    trainer.train(interrupted,rows,{},config,qualification,2,resume=True)
    a=torch.load(baseline/'model.pt',weights_only=True)['state_dict']
    b=torch.load(interrupted/'model.pt',weights_only=True)['state_dict']
    assert a.keys()==b.keys() and all(torch.equal(a[key],b[key]) for key in a)
    logs=[[json.loads(line) for line in (path/'updates.jsonl').read_text().splitlines()] for path in (baseline,interrupted)]
    fields=('step','epoch','ids','loss','gradient_norm','nonpadding_target_tokens')
    assert [[{key:row[key] for key in fields} for row in log] for log in logs][0]==[
        {key:row[key] for key in fields} for row in logs[1]]
    reports=[json.loads((path/'report.json').read_text()) for path in (baseline,interrupted)]
    assert reports[0]['epochs']==reports[1]['epochs']
