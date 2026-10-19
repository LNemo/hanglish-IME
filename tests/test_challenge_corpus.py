import hashlib
import importlib.util
import json
from pathlib import Path
import re


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "challenge"
SCRIPT = ROOT / "scripts" / "build_challenge_corpus.py"
FIELDS = {"id", "input", "context", "expected", "category", "family", "split", "provenance",
          "context_role", "boundary"}
CATEGORIES = {
    "copula_assimilation", "nasal_liquid_assimilation", "liaison", "palatalization",
    "homophone_context", "ambiguous_multi", "english_particle", "protected_token",
    "tense_consonants", "vowel_sequences",
}


def load(split):
    return [json.loads(line) for line in (DATA / f"{split}.jsonl").read_text(encoding="utf-8").splitlines()]


def builder_module():
    spec = importlib.util.spec_from_file_location("challenge_builder", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_committed_files_match_deterministic_builder():
    module = builder_module()
    for split in ("train", "dev", "test"):
        assert load(split) == module.build_split(split)


def test_schema_ids_and_unique_cases():
    expected_counts = {"train": 1820, "dev": 140, "test": 1000}
    for split, count in expected_counts.items():
        rows = load(split)
        assert len(rows) == count
        assert len({item["id"] for item in rows}) == count
        assert len({(item["input"], item["context"]) for item in rows}) == count
        for index, item in enumerate(rows, 1):
            assert set(item) == FIELDS
            assert item["id"] == f"{split}-{index:04d}"
            assert item["split"] == split
            assert item["category"] in CATEGORIES
            assert not item["family"].startswith(("train:", "dev:", "test:"))
            assert f"authored-{split}-" in item["provenance"]
            assert item["context_role"] in {"none", "semantic_hint", "non_decisive_hint"}
            assert item["boundary"] in {"word_spaced", "latin_suffix", "protected_backtick",
                                        "isolated_pronunciation", "stem_fragment"}
            assert item["input"] and isinstance(item["context"], str)
            assert item["expected"] and all(isinstance(value, str) and value for value in item["expected"])
            assert not re.search(r"[가-힣]", item["input"]), item


def test_test_set_is_balanced_across_ten_hard_categories():
    rows = load("test")
    counts = {category: sum(item["category"] == category for item in rows) for category in CATEGORIES}
    assert counts == {category: 100 for category in CATEGORIES}


def test_context_template_families_are_disjoint_between_splits():
    families = {split: {item["family"] for item in load(split)} for split in ("train", "dev", "test")}
    assert families["train"].isdisjoint(families["dev"])
    assert families["train"].isdisjoint(families["test"])
    assert families["dev"].isdisjoint(families["test"])


def test_split_surfaces_are_disjoint_and_semantic_hints_are_direct():
    rows = {split: load(split) for split in ("train", "dev", "test")}
    for left, right in (("train", "dev"), ("train", "test"), ("dev", "test")):
        assert {(item["input"], item["context"]) for item in rows[left]}.isdisjoint(
            {(item["input"], item["context"]) for item in rows[right]})
        left_contexts = {item["context"] for item in rows[left] if item["context"]}
        right_contexts = {item["context"] for item in rows[right] if item["context"]}
        assert left_contexts.isdisjoint(right_contexts)
    for split in rows:
        hints = [item for item in rows[split] if item["category"] == "homophone_context"]
        assert all(item["context_role"] == "semantic_hint" and item["context"].endswith(".") for item in hints)
        assert len({item["context"] for item in hints}) == len(hints)
        assert all(item["expected"][0] not in item["context"] for item in hints)


def test_ambiguous_and_disambiguated_policies_are_both_present():
    rows = {split: load(split) for split in ("train", "test")}
    for split in rows:
        ambiguous = [item for item in rows[split] if item["category"] == "ambiguous_multi"]
        contextual = [item for item in rows[split] if item["category"] == "homophone_context"]
        assert ambiguous and all(len(item["expected"]) >= 2 and item["context"] for item in ambiguous)
        assert contextual and all(len(item["expected"]) == 1 and item["context"] for item in contextual)
    # The ranker sees the spelling alternatives in train, but evaluation uses
    # wholly different semantic wrappers and recording scenarios.
    train_alternatives = {(item["input"], tuple(item["expected"])) for item in rows["train"]
                          if item["category"] == "ambiguous_multi"}
    test_alternatives = {(item["input"], tuple(item["expected"])) for item in rows["test"]
                         if item["category"] == "ambiguous_multi"}
    assert train_alternatives == test_alternatives
    assert ("nat", ("낫", "낮", "낯", "낟", "낱")) in test_alternatives
    assert ("got", ("곧", "곳", "곶")) in test_alternatives


def test_required_failure_modes_have_concrete_sentinels():
    rows = load("test")
    sources = {item["input"] for item in rows}
    assert "jeoneun haksaengimnida" in sources
    assert "jeoneun seoryureul geomtohamnida" in sources
    assert "oneul hangugeoreul baeunda" in sources
    assert "oneul gungmureul meongneunda" in sources
    assert "gachi georeotda" in sources
    assert "GraphRAGreul imnyeokaetda" in sources
    assert "`https://example.com/api/v1`reul geudaero boksahaetda" in sources
    assert "oneul kkocheul satda" in sources
    assert "oneul uijae anjatda" in sources
    assert any(item["input"] == "nat" and item["expected"] == ["낫", "낮", "낯", "낟", "낱"]
               for item in rows)


def test_audited_rr_and_held_out_gachi_probes():
    all_rows = {split: load(split) for split in ("train", "dev", "test")}
    pairs = {(item["input"], tuple(item["expected"])) for rows in all_rows.values() for item in rows}
    assert any("eumageul deunneunda" in source and expected[0].endswith("음악을 듣는다")
               for source, expected in pairs)
    assert any("kkocheul satda" in source and expected[0].endswith("꽃을 샀다")
               for source, expected in pairs)
    assert any("uiroereul badatda" in source and expected[0].endswith("의뢰를 받았다")
               for source, expected in pairs)
    assert any("imnyeongman heoyonghanda" in source and expected[0].endswith("입력만 허용한다")
               for source, expected in pairs)
    assert any("beobineul hwaginhanda" in source and expected[0].endswith("법인을 확인한다")
               for source, expected in pairs)
    assert any("byeohulchireul sseunda" in source and expected[0].endswith("벼훑이를 쓴다")
               for source, expected in pairs)
    all_sources = {source for source, _ in pairs}
    for rejected in ("sijakhamnida", "silseuphamnida", "yeonseuphamnida", "yaksokhanda",
                     "imnyeokhaetda", "eongeuphaetda", "girokhaetda", "bakkwiji"):
        assert not any(rejected in source for source in all_sources)
    held_out = {
        ("친구와 함께 가려고 한다.", "같이"),
        ("물건의 가격과 쓸모를 평가하고 있다.", "가치"),
    }
    assert held_out <= {(item["context"], item["expected"][0]) for item in all_rows["dev"]}
    assert held_out.isdisjoint({(item["context"], item["expected"][0]) for item in all_rows["train"]})


def test_mixed_script_literals_are_preserved_exactly():
    module = builder_module()
    terms = set(module.ENGLISH_TERMS)
    for split in ("train", "dev", "test"):
        for item in load(split):
            if item["category"] == "english_particle":
                term = next(term for term in terms if item["input"].startswith(term))
                assert item["expected"][0].startswith(term)
            elif item["category"] == "protected_token":
                source_literal = item["input"].split("`", 2)[1]
                target_literal = item["expected"][0].split("`", 2)[1]
                assert source_literal == target_literal


def test_bound_forms_are_never_presented_as_standalone_words():
    bound_forms = {"맞", "갔", "닮", "앓", "싫", "없", "앉", "많"}
    for split in ("train", "dev", "test"):
        for item in load(split):
            if bound_forms.intersection(item["expected"]):
                assert item["boundary"] == "stem_fragment"
                assert item["context_role"] in {"semantic_hint", "non_decisive_hint"}
                assert "어간" in item["context"] or item["category"] == "homophone_context"
    forbidden_ambiguous = {"맞", "갔", "닮", "앓", "싫", "없", "앉", "많"}
    assert all(forbidden_ambiguous.isdisjoint(item["expected"])
               for split in ("train", "dev", "test")
               for item in load(split) if item["category"] == "ambiguous_multi")


def test_train_lexicon_is_derived_only_from_exact_train_bytes():
    train_bytes = (DATA / "train.jsonl").read_bytes()
    lexical = json.loads((DATA / "train_lexicon.json").read_text(encoding="utf-8"))
    expected_counts = {}
    for item in load("train"):
        for train_text in [item["context"], *item["expected"]]:
            for word in re.findall(r"[가-힣]+", train_text):
                expected_counts[word] = expected_counts.get(word, 0) + 1
    assert lexical["train_sha256"] == hashlib.sha256(train_bytes).hexdigest()
    assert lexical["provenance"]["source_split"] == "train"
    assert lexical["entries"] == [
        {"text": word, "count": count} for word, count in sorted(expected_counts.items())
    ]


def test_builder_does_not_import_the_converter_under_evaluation():
    source = SCRIPT.read_text(encoding="utf-8")
    assert "hanglish.core" not in source
    assert "WorkbenchEngine" not in source


def test_manifest_hashes_the_frozen_corpus_files():
    manifest = json.loads((DATA / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["schema_version"] == 2
    assert manifest["corpus_version"] == "2.1"
    assert manifest["counts"] == {"train": 1820, "dev": 140, "test": 1000}
    assert "complete standard-dictionary headwords" in manifest["ambiguity_policy"]
    for name, expected_hash in manifest["sha256"].items():
        assert hashlib.sha256((DATA / name).read_bytes()).hexdigest() == expected_hash
