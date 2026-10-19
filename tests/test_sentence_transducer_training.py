from dataclasses import asdict
import json

import pytest

from hanglish.models.sentence_transducer import SentenceTransducerConfig
from scripts.train_sentence_transducer import batches, qualify


def evidence():
    config=SentenceTransducerConfig()
    steps=[dict(step=i+1, kind='length_bin',measured_length=length,
                loss=1.,gradient_norm=2.,elapsed_s=.1,rows=16,source_tokens=length,
                target_tokens=length,nonpadding_target_tokens=100) for i,length in enumerate((128,256,384,512))]
    return dict(completed=True,phase='preflight',pins={},config=asdict(config),rows=1,
                reload_max_logit_delta=0,peak_rss_bytes=1000000,elapsed_s=10,updates=steps,
                generation_probe=dict(kind='forced_no_eos_maximum_shape',batch_size=8,
                    source_tokens=512,max_generated_tokens=512,elapsed_s=1,
                    reasons=['token_limit']*8,generated_tokens=[512]*8))


@pytest.mark.parametrize('mutation', ['missing_generation','slow_generation','wrong_shape','wrong_kind','nonfinite_loss','nonfinite_memory','changed_pins'])
def test_qualification_rejects_unmeasured_or_mismatched_work(tmp_path,mutation):
    report=evidence()
    if mutation=='missing_generation':del report['generation_probe']
    elif mutation=='slow_generation':report['generation_probe']['elapsed_s']=100
    elif mutation=='wrong_shape':report['updates'][-1]['source_tokens']=20
    elif mutation=='wrong_kind':report['updates'][-1]['kind']='representative'
    elif mutation=='nonfinite_loss':report['updates'][0]['loss']=float('nan')
    elif mutation=='nonfinite_memory':report['peak_rss_bytes']=float('nan')
    else:report['pins']={'different':'input'}
    path=tmp_path/'report.json';path.write_text(json.dumps(report))
    with pytest.raises(ValueError):qualify(path,[{}],{},SentenceTransducerConfig())


def test_length_bucketing_keeps_every_training_row_once_per_epoch():
    rows=[dict(id=str(i),input='roman sentence '+('x'*i),expected=['한글 문장']) for i in range(270)]
    for epoch in (0,1):
        grouped=batches(rows,epoch)
        assert sorted(row['id'] for group in grouped for row in group)==sorted(row['id'] for row in rows)
        assert all(1<=len(group)<=16 for group in grouped)
        assert grouped==batches(rows,epoch)


def test_estimate_charges_upper_bins_monotonically_for_each_epoch(monkeypatch):
    from scripts import train_sentence_transducer as module
    def row(length):
        return dict(input='a '+'b'*(length-4),expected=['한 글'])
    values=[row(length) for length in (128,129,257,385)]
    schedules=[[[value] for value in values],[[values[0],values[-1]],[values[0]]]]
    monkeypatch.setattr(module,'batches',lambda rows,epoch:schedules[epoch])
    report=evidence()
    for step,seconds in zip(report['updates'],(.4,.2,.6,.5)):
        step['elapsed_s']=seconds
    # Monotone upper-bin costs .4,.4,.6,.6; partial batches are not discounted.
    assert module.estimate_training_seconds(report,values,2)==pytest.approx(3.75)
