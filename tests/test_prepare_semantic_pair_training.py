import json

import pytest

import scripts.prepare_semantic_pair_training as module


class FakeTokenizer:
    def encode(self, text, *, add_special_tokens, truncation):
        assert add_special_tokens is True
        assert truncation is False
        return list(range(len(text) + 2))


def candidate(source, text, score=1.0, kind="morphology"):
    return {"text": text, "score": score, "segments": [{
        "input_start": 0, "input_end": len(source), "source": source,
        "text": text, "kind": kind,
    }]}


def row(identity, group="group", source="imnida", context="",
        current="임니다", alternative="입니다", expected="SECRET_LABEL"):
    return {
        "id": identity, "group_id": group, "input": source,
        "context": context, "result": current, "expected": [expected],
        "accepted_positive_indices": [1], "label_aligned": True,
        "candidates": [
            candidate(source, current, 2.0),
            candidate(source, alternative, 1.0),
            candidate(source, source, None, "raw"),
        ],
    }


def envelope(current, alternative):
    metadata = {"id": "r", "input": "imnida", "context": "ctx",
                "fold": 2, "component": "component"}
    return module.build_pair_envelope(
        0, metadata, 0, 1, current, alternative, FakeTokenizer()
    )


def test_logical_swap_exchanges_roles_but_preserves_canonical_views():
    current = candidate("imnida", "임니다", 2.0)
    alternative = candidate("imnida", "입니다", 1.0)
    left = envelope(current, alternative)
    right = envelope(alternative, current)
    assert left["neural_views"] == right["neural_views"]
    assert left["neural_views"][0] != left["neural_views"][1]
    assert left["current"] == right["alternative"]
    assert left["alternative"] == right["current"]
    assert left["canonical_orientation"]["current_is_a"] is not right[
        "canonical_orientation"]["current_is_a"]


def test_neural_views_contain_only_role_free_text_contract():
    value = envelope(
        candidate("imnida", "임니다", 2.0),
        candidate("imnida", "입니다", 1.0),
    )
    for view in value["neural_views"]:
        payload = json.loads(view)
        assert set(payload) == {
            "roman", "context", "candidate_a_text", "candidate_b_text"
        }
        assert not ({"current", "alternative", "role", "label", "expected"}
                    & set(payload))


def test_records_keep_all_rows_without_saving_labels(monkeypatch):
    monkeypatch.setattr(module, "proposal_allowed", lambda *_: True)
    natural = [row("n-pair"), row("n-none", alternative="imnida")]
    challenge = [row("c-pair", group="challenge")]
    monkeypatch.setattr(
        module, "proposal_allowed",
        lambda source, current, alternative: alternative["text"] != source,
    )
    rows, pairs = module.build_records(natural, challenge, FakeTokenizer())
    assert len(rows) == 3
    assert rows[1]["pair_indices"] == []
    assert [value["index"] for value in rows] == [0, 1, 2]
    assert all(pair["current_source_index"] == 0 for pair in pairs)
    serialized = json.dumps({"rows": rows, "pairs": pairs}, ensure_ascii=False)
    assert "SECRET_LABEL" not in serialized
    assert "expected" not in serialized and "label_aligned" not in serialized


def test_fold_closure_unites_same_group_and_normalized_source(monkeypatch):
    monkeypatch.setattr(module, "proposal_allowed", lambda *_: False)
    natural = [
        row("a", group="document", source="one", context="x"),
        row("b", group="document", source="two", context="y"),
    ]
    challenge = [
        row("c", group="other", source=" one ", context="x"),
        row("d", group="other", source="four", context="z"),
    ]
    rows, pairs = module.build_records(natural, challenge, FakeTokenizer())
    assert pairs == []
    assert len({(value["fold"], value["component"]) for value in rows}) == 1


def test_token_lengths_are_measured_without_truncation():
    views = ["x" * 510, "y" * 511]
    lengths = module.tokenize_views(FakeTokenizer(), views)
    assert lengths == [512, 513]


def test_source_pins_include_transitive_romanization_policy():
    loaded = {
        "manifest_raw": b"manifest", "cases_raw": b"cases",
        "selection_raw": b"selection",
    }
    pins = module.source_pins(loaded, loaded)
    path = module.ROOT / "hanglish/core/phonology.py"
    assert "hanglish/core/phonology.py" in module.DEPENDENCIES
    assert pins["hanglish/core/phonology.py"] == module.file_digest(path)


def test_prepare_refuses_existing_output(tmp_path):
    with pytest.raises(FileExistsError):
        module.prepare(tmp_path, tokenizer_loader=FakeTokenizer)
