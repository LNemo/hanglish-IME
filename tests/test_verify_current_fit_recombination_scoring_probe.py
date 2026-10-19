from copy import deepcopy
from datetime import datetime, timezone
import importlib.util
from pathlib import Path

import pytest

from scripts import current_recombination_scoring as scorer
from scripts import verify_current_fit_recombination_scoring_probe as verifier
_spec = importlib.util.spec_from_file_location(
    "scoring_fixture", Path(__file__).with_name("test_current_recombination_scoring.py"))
_fixtures = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_fixtures)
fixture, Ranker = _fixtures.fixture, _fixtures.Ranker


def captured_fixture():
    original, census = fixture()
    scoring = scorer.score_row(original, census, Ranker(), check_original_parity=True)
    captured = {"id": original["id"], "component_id": original["component_id"],
                "request": original["request"],
                "source_shard": census["source_reference"]["source_shard"],
                "source_row_index": census["source_reference"]["source_row_index"],
                "scoring": scoring}
    return original, census, captured


def test_recomputes_features_and_preserves_cached_original_scores():
    original, census, captured = captured_fixture()
    assert verifier.verify_scoring_row(original, census, captured) == {
        "original": 2, "novel": 1, "scored_novel": 1, "unsupported_novel": 0}
    assert captured["scoring"]["original"][0]["sentence_score"] == 2.0


def test_full_capture_requires_explicitly_absent_original_replay():
    original, census, captured = captured_fixture()
    with pytest.raises(ValueError, match="unperformed"):
        verifier.verify_scoring_row(original,census,captured,original_parity_required=False)
    captured["scoring"]["original_score_parity"] = None
    assert verifier.verify_scoring_row(original,census,captured,original_parity_required=False)["novel"] == 1
    with pytest.raises(ValueError, match="parity"):
        verifier.verify_scoring_row(original,census,captured)


@pytest.mark.parametrize("mutation,match", [
    (lambda o,c,r: r["scoring"].update(fallback_public=[{"text":"unexpected"}]), "fallback"),
    (lambda o,c,r: r["scoring"].update(ranked=False), "ranked"),
    (lambda o,c,r: r["scoring"].update(score_texts=[]), "scored text"),
    (lambda o, c, r: r["scoring"]["original"][0].update(features=[0, 0, 0, 0]),
     "original cached score or feature"),
    (lambda o, c, r: r["scoring"]["novel"][0].update(sentence_score=None),
     "score/status"),
    (lambda o, c, r: r.update(source_row_index=99), "source location"),
    (lambda o, c, r: r["scoring"]["original_score_parity"].update(max_absolute_difference=2e-5),
     "parity"),
])
def test_rejects_feature_score_status_source_and_parity_mutations(mutation, match):
    original, census, captured = captured_fixture(); mutation(original, census, captured)
    with pytest.raises(ValueError, match=match):
        verifier.verify_scoring_row(original, census, captured)


def resource(stage, operation, elapsed, operation_elapsed=.01):
    return {"at": datetime.now(timezone.utc).isoformat(), "stage": stage,
            "operation": operation,
            "operation_elapsed_s": operation_elapsed if operation else None,
            "elapsed_s": elapsed, "rss_bytes": 100,
            "swap_used_bytes": 1000, "swap_growth_bytes": 0}


def trace(operations):
    rows=[]; elapsed=0.
    for index,operation in enumerate(operations):
        update = operation.startswith("score:") or operation.startswith("write_")
        duration = .01 if update else None
        elapsed += .1; rows.append(resource("operation_start",operation,elapsed,duration))
        elapsed += .1; rows.append(resource("operation_end",operation,elapsed,duration))
        if index == len(operations)-2:
            elapsed += .1; rows.append(resource("after_final_rehash",None,elapsed))
    elapsed += .1; rows.append(resource("after_completion",None,elapsed))
    return rows


def test_resource_replay_distinguishes_unbounded_initialization_from_scoring_limit():
    operations=["authenticate_sources","load_frozen_sentence_ranker","score:id",
                "write_closure","write_report","final_rehash","write_completion"]
    rows=trace(operations)
    assert verifier.verify_resources(rows,operations)["operations"] == len(operations)
    bad=deepcopy(rows); bad[4]["operation_elapsed_s"]=31.
    with pytest.raises(ValueError,match="30 seconds"):
        verifier.verify_resources(bad,operations)


def test_resource_replay_rejects_missing_scoring_clock_and_fabricated_initialization_clock():
    operations=["authenticate_sources","load_frozen_sentence_ranker","score:id",
                "write_closure","write_report","final_rehash","write_completion"]
    for index,value in [(4,None),(0,100.)]:
        rows=trace(operations);rows[index]["operation_elapsed_s"]=value
        with pytest.raises(ValueError,match="clock|duration"):
            verifier.verify_resources(rows,operations)


def test_projection_formula_rejects_changed_value():
    rows=[{"id":f"id-{i}","input":"a"*(i+1)} for i in range(16)]
    groups=verifier.probe_groups(rows)
    timings={group["probe_id"]:float(i+1) for i,group in enumerate(groups)}
    shards=[[row["id"] for row in rows[i::6]] for i in range(6)]
    by_id={identifier:timings[group["probe_id"]] for group in groups for identifier in group["ids"]}
    expected=[1.3*(2+sum(by_id[item] for item in shard))+60 for shard in shards]
    assert verifier.verify_projection(groups,shards,timings,2.,expected,True)==expected
    changed=list(expected); changed[0] += .1
    with pytest.raises(ValueError,match="projection"):
        verifier.verify_projection(groups,shards,timings,2.,changed,True)
