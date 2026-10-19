import copy
from types import SimpleNamespace

import pytest

from scripts.prepare_natural_word_ranker import add_targets, capture_groups, target_spans, validate_case


def option(text, score=7., kind='lexicon'):
    return SimpleNamespace(text=text, score=score, kind=kind)


def test_repeated_homophones_align_by_position_and_preserve_missing_targets():
    source = {'id': 'r', 'input': 'gachi gachi', 'expected': ['같이 가치'],
              'origin': 'natural', 'split': 'train', 'group_id': 'doc', 'context': ''}
    pieces = [('gachi', 0, [option('가치'), option('까치')]),
              (' ', 5, [option(' ', 0.)]),
              ('gachi', 6, [option('가치'), option('같이'), option('gachi', 0., 'raw')])]
    records = [('  [MASK]  가치', '가치', 1.), ('  [MASK]  가치', '까치', 0.),
               (' 가치  [MASK] ', '가치', 1.), (' 가치  [MASK] ', '같이', 0.)]
    groups = capture_groups(pieces, records, '')
    assert all('expected' not in group for group in groups)
    groups = add_targets(source, groups)
    assert groups[0]['expected'] == ['같이'] and groups[0]['positive_indices'] == []
    assert groups[1]['expected'] == ['가치'] and groups[1]['positive_indices'] == [0]
    assert groups[0]['id'] != groups[1]['id']
    validate_case({**source, 'groups': groups}, source)
    wrong = copy.deepcopy(records)
    wrong[0] = ('gold context [MASK]', '가치', 1.)
    with pytest.raises(ValueError, match='runtime grouping'):
        capture_groups(pieces, wrong, '')
    with pytest.raises(ValueError, match='unconsumed'):
        capture_groups(pieces, records + [records[0]], '')


def test_forward_alignment_preserves_digits_and_english():
    assert target_spans('12ire testreul', ['12일에 test를']) == {(2, 5): {'일에'}, (6, 14): {'test를'}}
    assert target_spans('gachi', ['같이', '가치']) == {(0, 5): {'같이', '가치'}}
    with pytest.raises(ValueError, match='exact forward'):
        target_spans('gachi', ['다른 말'])


@pytest.mark.parametrize('field,value', [('positive_indices', [True]), ('positive_indices', None),
    ('teacher_scores', [float('nan'), 0.]), ('base_scores', [True, 0.]),
    ('candidates', ['가치', '가치']), ('start', True), ('expected', ['같이'])])
def test_group_validation_rejects_corrupt_features_and_labels(field, value):
    source = {'id': 'r', 'input': 'gachi', 'expected': ['가치'], 'origin': 'natural',
              'split': 'train', 'group_id': 'doc', 'context': ''}
    group = {'id': 'r:0:5', 'source': 'gachi', 'start': 0, 'end': 5, 'context': ' [MASK] ',
             'expected': ['가치'], 'candidates': ['가치', '같이'], 'positive_indices': [0],
             'base_scores': [7., 7.], 'teacher_scores': [1., 0.], 'kinds': ['lexicon', 'lexicon']}
    group[field] = value
    with pytest.raises(ValueError):
        validate_case({**source, 'groups': [group]}, source)


def test_bounded_training_preserves_all_positives_and_hard_negatives():
    from scripts.train_natural_word_ranker import bounded_example, RECIPE
    group = {'id': 'r', 'context': 'runtime [MASK]', 'candidates': ['같이', '가치', '까치', '가찌'],
             'positive_indices': [0, 1], 'base_scores': [1., 1., 20., 1.],
             'teacher_scores': [0., 0., 0., 1.]}
    example = bounded_example('retention', group, {**RECIPE, 'candidate_limit': 3})
    assert example.candidates == ('같이', '가치', '까치')
    assert example.positives == (0, 1)
    assert example.context == group['context']
    assert bounded_example('natural', {**group, 'positive_indices': []}, RECIPE) is None


def test_group_sampling_is_independent_of_labels_and_input_order():
    from scripts.train_natural_word_ranker import select_groups, RECIPE
    rows = [{'origin': 'natural', 'groups': [{'id': str(i), 'positive_indices': []} for i in range(10)]}]
    changed = copy.deepcopy(rows)
    changed[0]['groups'].reverse()
    for group in changed[0]['groups']: group['positive_indices'] = [0]
    recipe = {**RECIPE, 'source_group_limit': 4}
    assert [g['id'] for _,g in select_groups(rows, recipe)] == [g['id'] for _,g in select_groups(changed, recipe)]


def test_evaluation_keeps_single_group_calls_and_static_bias():
    from scripts.train_natural_word_ranker import evaluate_groups
    class Ranker:
        def score_pairs(self, contexts, candidates):
            assert candidates == ['single', 'a', 'b']
            return [9., 1., 0.]
    rows = [{'groups': [{'id': 'single', 'context': '[MASK]', 'candidates': ['single'],
                       'base_scores': [0.], 'teacher_scores': [9.], 'positive_indices': [0]},
                      {'id': 'multiple', 'context': '[MASK]', 'candidates': ['a', 'b'],
                       'base_scores': [0., 10.], 'teacher_scores': [1., 0.], 'positive_indices': [1]}]}]
    result = evaluate_groups(Ranker(), rows, check_teacher=True)
    assert result['ambiguous_groups'] == 1 and result['correct'] == 1
    assert result['outcomes'][0]['result'] == 'b'
    rows[0]['groups'][0]['teacher_scores'] = [8.]
    with pytest.raises(ValueError, match='teacher differs'):
        evaluate_groups(Ranker(), rows, check_teacher=True)


def test_word_loss_matches_runtime_bias_and_retains_valid_teacher_preferences():
    import torch
    from scripts.train_natural_word_ranker import WordExample, word_loss, RECIPE
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.scores = torch.nn.Parameter(torch.zeros(3))
        def forward(self, input_ids):
            return SimpleNamespace(logits=self.scores[input_ids[:, 0]].unsqueeze(1))
    def tokenizer(candidates, contexts, **kwargs):
        return {'input_ids': torch.arange(len(candidates)).unsqueeze(1)}
    model = Model()
    example = WordExample('[MASK]', ('a', 'b', 'c'), (0, 2), (0., 10., 0.), (0., 0., 0.), 'n', 'natural')
    loss = word_loss(model, tokenizer, [example], 96, RECIPE)
    expected = torch.logsumexp(torch.tensor([0., 1.5, 0.]), 0) - torch.log(torch.tensor(2.))
    assert torch.allclose(loss, expected)
    loss.backward()
    assert model.scores.grad[0] < 0 and model.scores.grad[2] < 0 and model.scores.grad[1] > 0
    model.zero_grad()
    example = WordExample('[MASK]', ('a', 'b', 'c'), (0, 1, 2), (0., 0., 0.), (0., 2., 0.), 'r', 'retention')
    word_loss(model, tokenizer, [example], 96, RECIPE).backward()
    assert model.scores.grad[1] < 0 and model.scores.grad[0] > 0


def test_word_manifest_rejects_tampering_and_changed_runtime(monkeypatch, tmp_path):
    import json
    import scripts.prepare_natural_word_ranker as module
    source = {'id': 'empty', 'origin': 'natural', 'split': 'train', 'group_id': 'doc',
              'input': 'gachi', 'context': '', 'expected': ['가치']}
    sources = {role: [source] for role in module.ROLES}
    monkeypatch.setattr(module, 'validate_prepared', lambda *_: ({}, sources))
    monkeypatch.setattr(module, 'source_hashes', lambda: {'runtime': 'pinned'})
    outputs = {}
    for role in module.ROLES:
        path = tmp_path / (role+'.jsonl')
        path.write_text(json.dumps({**source, 'groups': []})+'\n')
        outputs[role] = {'sha256': module.file_sha256(path), 'bytes': path.stat().st_size, 'rows': 1}
    path = tmp_path / 'manifest.json'
    manifest = {'schema_version': 1, 'source_manifest_sha256': module.SOURCE_MANIFEST,
                'no_gold_injection': True, 'test_data_read': False, 'source_hashes': {'runtime': 'pinned'},
                'outputs': outputs}
    path.write_text(json.dumps(manifest)); pin = module.file_sha256(path)
    module.validate_word_data(tmp_path, pin)
    monkeypatch.setattr(module, 'source_hashes', lambda: {'runtime': 'changed'})
    with pytest.raises(ValueError, match='sources changed'):
        module.validate_word_data(tmp_path, pin)
    monkeypatch.setattr(module, 'source_hashes', lambda: {'runtime': 'pinned'})
    target = tmp_path / 'natural_train.jsonl'
    target.write_text(target.read_text().replace('gachi', 'wrong'))
    with pytest.raises(ValueError, match='integrity'):
        module.validate_word_data(tmp_path, pin)
    manifest['outputs']['natural_train']['sha256'] = module.file_sha256(target)
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match='explicitly pinned'):
        module.validate_word_data(tmp_path, pin)


def test_final_evaluation_rejects_failed_development_gate(monkeypatch, tmp_path):
    import json
    import scripts.evaluate_natural_word_ranker as module
    checkpoint = tmp_path / 'checkpoint'; checkpoint.mkdir()
    (checkpoint / 'model.safetensors').write_text('fake fixture weights')
    digest = module.file_sha256(checkpoint / 'model.safetensors')
    (tmp_path / 'report.json').write_text(json.dumps({'checkpoint_sha256': digest,
                                                   'initial': {'correct': 1}, 'final': {'correct': 2}}))
    (tmp_path / 'runtime-dev.json').write_text(json.dumps({'passed': False, 'checkpoint_sha256': digest}))
    ranker = SimpleNamespace(metadata={'natural_word_preparation': {'path': str(tmp_path), 'manifest_sha256': 'pin'}})
    monkeypatch.setattr(module.SemanticRanker, 'load', lambda *_: ranker)
    monkeypatch.setattr(module, 'validate_word_data', lambda *_: ({}, {}))
    monkeypatch.setattr(module, 'source_hashes', lambda: {})
    with pytest.raises(ValueError, match='hybrid development gate'):
        module.evaluate(checkpoint, tmp_path / 'final.json', final=True)
    assert not (tmp_path / 'final.json').exists()
