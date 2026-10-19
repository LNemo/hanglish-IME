import math
import subprocess
import sys

import pytest

from hanglish.models.phonetic_choice_features import FEATURE_NAMES, project_row


def score(pair_id, source, target, mean):
    tokens = 3 * len(target) + 1
    return {
        "pair_id": pair_id,
        "source": source,
        "target": target,
        "log_probability": mean * tokens,
        "token_count": tokens,
        "mean_log_probability": mean,
    }


def edge(start, end, source, target, status, *, pair_id=None, match=None):
    value = {"input_start": start, "input_end": end, "source": source,
             "target": target, "status": status}
    if status == "supported":
        value.update(pair_id=pair_id, romanize_match=match)
    return value


def row(source, candidates, edges, paths, *, base_scores=None, positives=None):
    count = len(candidates)
    return {
        "schema": "phonetic-likelihood-fit-v1-row",
        "id": "row-1",
        "input": source,
        "component_id": "component-1",
        "candidates": candidates,
        "base_scores": list(base_scores if base_scores is not None else range(count)),
        "positive_indices": list(positives if positives is not None else [0]),
        "selected_indices": list(range(count)),
        "edges": edges,
        "candidate_paths": paths,
    }


def path(name, *indices):
    return {"path_id": name, "edge_indices": list(indices)}


def test_feature_names_are_stable_and_identifiable():
    assert FEATURE_NAMES == (
        "length_weighted_mean_jamo_eos_log_probability",
        "supported_source_fraction",
        "unsupported_conversion_source_fraction",
        "forward_mismatch_supported_source_fraction",
    )


def test_import_is_pure_and_does_not_load_torch_or_transducer():
    subprocess.run([
        sys.executable, "-c",
        "import sys; import hanglish.models.phonetic_choice_features; "
        "assert 'torch' not in sys.modules; "
        "assert 'hanglish.models.phonetic_transducer' not in sys.modules",
    ], check=True)


def test_source_length_weights_copy_neutrality_unsupported_and_mismatch():
    source = "ab cd"
    edges = [
        edge(0, 2, "ab", "가", "supported", pair_id="ab-ga", match=False),
        edge(2, 3, " ", " ", "copy"),
        edge(3, 5, "cd", "c디", "unsupported_shape"),
    ]
    value = project_row(
        row(source, ["가 c디"], edges, [[path("p", 0, 1, 2)]], base_scores=[-3.5]),
        {"ab-ga": score("ab-ga", "ab", "가", -2.0)},
    )
    assert value["features"] == [[-0.8, 0.4, 0.4, 0.4]]
    assert value["base_scores"] == [-3.5]


def test_copy_only_path_has_zero_vector_and_zero_weights_preserve_base_scores():
    frozen = [-4.0, -2.0]
    value = project_row(
        row("ab", ["ab", "가"], [
            edge(0, 2, "ab", "ab", "copy"),
            edge(0, 2, "ab", "가", "supported", pair_id="ab-ga", match=True),
        ], [[path("copy", 0)], [path("convert", 1)]], base_scores=frozen),
        {"ab-ga": score("ab-ga", "ab", "가", -1.25)},
    )
    assert value["features"][0] == [0.0, 0.0, 0.0, 0.0]
    assert value["features"][1] == [-1.25, 1.0, 0.0, 0.0]
    zero = [0.0] * len(FEATURE_NAMES)
    rescored = [base + sum(weight * feature for weight, feature in zip(zero, vector))
                for base, vector in zip(value["base_scores"], value["features"])]
    assert rescored == frozen


def test_multiple_alignment_paths_use_uniform_arithmetic_feature_mean():
    edges = [
        edge(0, 2, "ab", "가나", "supported", pair_id="whole", match=True),
        edge(0, 1, "a", "가", "supported", pair_id="left", match=False),
        edge(1, 2, "b", "나", "supported", pair_id="right", match=True),
    ]
    value = project_row(
        row("ab", ["가나"], edges, [[path("whole-path", 0), path("split-path", 1, 2)]]),
        {
            "whole": score("whole", "ab", "가나", -1.0),
            "left": score("left", "a", "가", -2.0),
            "right": score("right", "b", "나", -4.0),
        },
    )
    # Whole path: [-1, 1, 0, 0]; split path: [-3, 1, 0, .5].
    assert value["features"] == [[-2.0, 1.0, 0.0, 0.25]]


@pytest.mark.parametrize("mutation,match", [
    (lambda item, scores: scores.clear(), "score is missing"),
    (lambda item, scores: scores["ab-ga"].update(source="wrong"), "score identity"),
    (lambda item, scores: scores["ab-ga"].update(token_count=3), "score value"),
    (lambda item, scores: scores["ab-ga"].update(log_probability=math.nan), "score value"),
    (lambda item, scores: item["candidate_paths"][0][0].update(edge_indices=[]), "path is empty"),
    (lambda item, scores: item["edges"][0].update(input_start=1), "source offset"),
    (lambda item, scores: item["candidate_paths"][0][0].update(edge_indices=[1]), "full source"),
    (lambda item, scores: item["candidates"].__setitem__(0, "나"), "target differs"),
    (lambda item, scores: item["base_scores"].__setitem__(0, math.inf), "candidate domain"),
])
def test_malformed_score_coverage_target_and_base_are_rejected(mutation, match):
    item = row("ab", ["가"], [
        edge(0, 2, "ab", "가", "supported", pair_id="ab-ga", match=True),
        edge(0, 1, "a", "a", "copy"),
    ], [[path("p", 0)]], base_scores=[-1.0])
    scores = {"ab-ga": score("ab-ga", "ab", "가", -1.0)}
    mutation(item, scores)
    with pytest.raises(ValueError, match=match):
        project_row(item, scores)


def test_empty_alternatives_and_duplicate_path_ids_are_rejected():
    copy_edge = edge(0, 2, "ab", "ab", "copy")
    empty = row("ab", ["ab"], [copy_edge], [[]])
    with pytest.raises(ValueError, match="no alignment"):
        project_row(empty, {})

    duplicate = row("ab", ["ab", "ab!"], [
        copy_edge,
        edge(0, 2, "ab", "ab!", "unsupported_shape"),
    ], [[path("same", 0)], [path("same", 1)]])
    with pytest.raises(ValueError, match="duplicated"):
        project_row(duplicate, {})
