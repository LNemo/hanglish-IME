import copy
import json
import pytest

from scripts.prepare_natural_ranker import prepare_row, validate_row, select_rows


def source():
    return {'id': 'n1', 'text': '함께 같이 일한다.', 'left_context': '동료와 협력한다.',
            'group_id': 'document1'}


def test_source_only_proposals_keep_uncovered_case_and_context():
    seen = []
    row = prepare_row(source(), 'natural_dev', lambda text, context: seen.append((text, context)) or ['다른 후보'])
    assert seen == [(row['input'], '동료와 협력한다.')]
    assert row['positive_indices'] == []
    assert row['candidates'] == ['다른 후보']
    validate_row(row, source(), 'natural_dev')


def test_retention_keeps_all_accepted_positives_without_injection():
    src = {'id': 'r1', 'input': 'gachi', 'context': '', 'family': 'homophones', 'expected': ['같이', '가치']}
    row = prepare_row(src, 'retention_train', lambda *_: ['가치', '까치', '같이', '가치'])
    assert row['candidates'] == ['가치', '까치', '같이']
    assert row['positive_indices'] == [0, 2]
    validate_row(row, src, 'retention_train')


@pytest.mark.parametrize('field,value', [('split', 'train'), ('expected', ['답 교체']),
                                      ('positive_indices', [99]), ('positive_indices', [True]),
                                      ('candidates', ['', '']), ('candidates', ['중복', '중복'])])
def test_rejects_relabeling_misplaced_dev_and_invalid_pool(field, value):
    src = source()
    row = prepare_row(src, 'natural_dev', lambda *_: [src['text'], '다른 후보'])
    row[field] = value
    with pytest.raises(ValueError):
        validate_row(row, src, 'natural_dev')


def test_selection_ignores_labels_and_source_order():
    rows = [{'id': str(i), 'expected': ['a']} for i in range(20)]
    chosen = select_rows(rows, 'natural_train', 5)
    changed = copy.deepcopy(rows[::-1])
    for row in changed:
        row['expected'] = ['different']
    assert [r['id'] for r in chosen] == [r['id'] for r in select_rows(changed, 'natural_train', 5)]


def test_prepared_guard_rejects_data_and_coordinated_manifest_tampering(monkeypatch, tmp_path):
    import scripts.prepare_natural_ranker as module
    src = source()
    retention = {'id': 'r1', 'input': 'gachi', 'context': '', 'family': 'f', 'expected': ['같이']}
    sources = {'natural_train': [src], 'natural_dev': [{**src, 'id': 'n2'}],
               'retention_train': [retention]}
    monkeypatch.setattr(module, 'selected_sources', lambda: (sources, {'verified': True}))
    monkeypatch.setattr(module, 'frozen_hashes', lambda: {'source': 'pinned'})
    outputs = {}
    for role, rows in sources.items():
        path = tmp_path / (role + '.jsonl')
        row = prepare_row(rows[0], role, lambda *_: ['같이', '다른 후보'])
        path.write_text(json.dumps(row, ensure_ascii=False) + '\n')
        outputs[role] = {'sha256': module.file_sha256(path), 'bytes': path.stat().st_size, 'rows': 1}
    manifest = {'schema_version': 1, 'selection_salt': module.SALT, 'limits': module.LIMITS,
                'no_gold_injection': True, 'test_data_read': False,
                'frozen_hashes': {'source': 'pinned'}, 'challenge_guard': {'verified': True},
                'outputs': outputs}
    path = tmp_path / 'manifest.json'
    path.write_text(json.dumps(manifest))
    pin = module.file_sha256(path)
    module.validate_prepared(tmp_path, pin)
    target = tmp_path / 'natural_dev.jsonl'
    target.write_text(target.read_text().replace('다른 후보', '다른 정답'))
    with pytest.raises(ValueError, match='integrity'):
        module.validate_prepared(tmp_path, pin)
    manifest['outputs']['natural_dev']['sha256'] = module.file_sha256(target)
    manifest['outputs']['natural_dev']['bytes'] = target.stat().st_size
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match='explicitly pinned'):
        module.validate_prepared(tmp_path, pin)


def test_train_candidate_cap_keeps_all_positives_and_skips_missing():
    from scripts.train_natural_sentence_ranker import bounded_example
    row = {'id': 'a', 'origin': 'natural', 'context': '',
           'candidates': [str(i) for i in range(12)], 'positive_indices': [9, 11]}
    example = bounded_example(row, limit=4)
    assert example.candidates == ('9', '11', '0', '1')
    assert example.positives == (0, 1)
    assert bounded_example({**row, 'positive_indices': []}) is None


def test_balance_retains_both_sources_deterministically():
    from scripts.train_natural_sentence_ranker import balanced_schedule
    result = balanced_schedule(['n1'], ['r1', 'r2', 'r3'], 5)
    assert len(result) == 6 and result[::2] == ['n1'] * 3
    assert set(result[1::2]) == {'r1', 'r2', 'r3'}
    assert result == balanced_schedule(['n1'], ['r1', 'r2', 'r3'], 5)


def test_runtime_evaluation_keeps_uncovered_rows_and_preserves_fallback():
    from scripts.train_natural_sentence_ranker import evaluate_runtime_pool
    class Ranker:
        def can_score(self, text):
            return text != 'too long'
        def score_pairs(self, contexts, candidates):
            return [0., 1.]
    rows = [{'id': 'missing', 'input': 'two words', 'context': '',
             'candidates': ['a', 'b'], 'positive_indices': []},
            {'id': 'length', 'input': 'two words', 'context': '',
             'candidates': ['right', 'too long'], 'positive_indices': [0]},
            {'id': 'isolated', 'input': 'word', 'context': '',
             'candidates': ['first', 'second'], 'positive_indices': [0]}]
    report = evaluate_runtime_pool(Ranker(), rows)
    assert report['rows'] == 3 and report['covered'] == 2 and report['correct'] == 2
    assert report['fallbacks'] == 2


def test_natural_dev_target_cannot_leak_through_retention_source():
    from scripts.train_natural_sentence_ranker import validate_mixed_split
    data = {'natural_train': [{'id': 'n', 'context': '', 'expected': ['자연 학습 문장']}],
            'retention_train': [{'id': 'r', 'context': '', 'expected': ['기존 학습 문장']}],
            'natural_dev': [{'expected': ['개발 검증 문장']}]}
    assert validate_mixed_split(data)['dev_target_in_mixed_train_overlap_rows'] == 0
    data['retention_train'][0]['context'] = '앞 문장: 개발 검증 문장'
    with pytest.raises(ValueError, match='mixed training'):
        validate_mixed_split(data)


def test_training_loss_improves_both_accepted_candidates():
    import torch
    from types import SimpleNamespace
    from scripts.train_sentence_ranker import SentenceExample, _loss
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.scores = torch.nn.Parameter(torch.zeros(3))
        def forward(self, input_ids):
            return SimpleNamespace(logits=self.scores[input_ids[:, 0]].unsqueeze(1))
    def tokenizer(sentences, contexts=None, **kwargs):
        if contexts is None:
            return {'input_ids': list(range(len(sentences)))}
        return {'input_ids': torch.arange(len(sentences)).unsqueeze(1)}
    tokenizer.num_special_tokens_to_add = lambda pair: 3
    model = Model()
    example = SentenceExample('문맥', ('같이', '까치', '가치'), (0, 2), 'multi', 'retention')
    optimizer = torch.optim.SGD(model.parameters(), lr=.1)
    loss = _loss(model, tokenizer, [example], SimpleNamespace(max_length=96, device='cpu'))
    loss.backward(); optimizer.step()
    assert model.scores[0] > model.scores[1] and model.scores[2] > model.scores[1]
    assert torch.equal(model.scores[0], model.scores[2])


def test_distillation_retains_teacher_preference_when_all_answers_are_valid():
    import torch
    from types import SimpleNamespace
    from scripts.train_sentence_ranker import SentenceExample
    from scripts.train_natural_sentence_ranker import distillation_loss, HEAD_RECIPE
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.scores = torch.nn.Parameter(torch.zeros(3))
        def forward(self, input_ids):
            return SimpleNamespace(logits=self.scores[input_ids[:, 0]].unsqueeze(1))
    def tokenizer(sentences, contexts=None, **kwargs):
        if contexts is None:
            return {'input_ids': list(range(len(sentences)))}
        return {'input_ids': torch.arange(len(sentences)).unsqueeze(1)}
    tokenizer.num_special_tokens_to_add = lambda pair: 3
    model = Model()
    example = SentenceExample('', ('a', 'b', 'c'), (0, 1, 2), 'r', 'retention')
    args = SimpleNamespace(max_length=96, device='cpu')
    loss = distillation_loss(model, tokenizer, [example], args, {'r': [0., 2., 0.]}, HEAD_RECIPE)
    loss.backward()
    assert model.scores.grad[1] < 0
    assert model.scores.grad[0] > 0 and model.scores.grad[2] > 0
    with pytest.raises(ValueError, match='teacher candidate'):
        distillation_loss(model, tokenizer, [example], args, {'r': [1.]}, HEAD_RECIPE)
