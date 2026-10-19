"""Integration checks for actual imported trainer APIs before model loading."""
import ast
from pathlib import Path

from scripts import train_difference_pairwise_lora as trainer


def test_trainer_external_module_attributes_exist():
    modules = {name: getattr(trainer, name) for name in ('system', 'shared', 'pairwise', 'evaluation')}
    tree = ast.parse(Path(trainer.__file__).read_text())
    referenced = {(node.value.id, node.attr) for node in ast.walk(tree)
                  if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name)
                  and node.value.id in modules}
    missing = [f'{module}.{name}' for module, name in sorted(referenced)
               if not hasattr(modules[module], name)]
    assert missing == [], f'trainer references absent runtime APIs: {missing}'


def test_checkpoint_uses_actual_host_reader_symbols(monkeypatch):
    captured = {}
    def observer(**kwargs):
        captured.update(kwargs)
        return ({'ok': True}, {'stage': kwargs['stage']})
    monkeypatch.setattr(trainer, 'observe_resources', observer)
    result = trainer.observe_training_resources(stage='before_model_load', mx=object(),
        started=0., baseline={'swap_used_bytes': 0})
    assert captured['rss_reader'] is trainer.system.rss_bytes
    assert captured['system_reader'] is trainer.system.system_resources
    assert result[1]['stage'] == 'before_model_load'
    assert isinstance(captured['rss_reader'](), int)
