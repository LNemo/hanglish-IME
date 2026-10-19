import json
from pathlib import Path
import shutil

import pytest

from scripts import train_sentence_ctc as engine
from scripts import train_sentence_ctc_growth as growth


def test_frozen_growth_inputs_and_7704_row_schedule():
    growth.verify_control()
    proof = growth.verify_growth_inputs()
    rows = engine.read_training(growth.EXTRA)
    growth.validate_rows(rows)
    assert proof["rows"] == 2048 and len(rows) == 7704
    assert len(rows) == growth.EXPECTED_BASE_ROWS + growth.EXPECTED_EXTRA_ROWS
    schedules = [engine.batches(rows, epoch) for epoch in range(engine.EPOCHS)]
    assert [len(values) for values in schedules] == [482] * 8
    assert sum(map(len, schedules)) == 3856
    assert all(1 <= len(batch) <= engine.BATCH
               for schedule in schedules for batch in schedule)
    assert [[row["id"] for row in batch] for batch in schedules[0]] == [
        [row["id"] for row in batch] for batch in engine.batches(rows, 0)]


def test_growth_pins_extend_engine_identity_with_wrapper_and_proof():
    pins = growth.growth_pins()
    expected = engine.input_pins(growth.EXTRA)
    assert all(pins[path] == digest for path, digest in expected.items())
    assert pins[str(Path(growth.__file__))] == engine.sha(Path(growth.__file__))
    assert pins[str(growth.PROOF)] == growth.EXPECTED_PROOF_SHA256


@pytest.mark.parametrize("name", ["train.jsonl", "manifest.json",
                                   "independent-verification.json"])
def test_tampered_growth_data_or_proof_is_rejected(tmp_path, name):
    extra = tmp_path / "extra"
    extra.mkdir()
    for source_name in ("train.jsonl", "manifest.json", "independent-verification.json"):
        shutil.copyfile(growth.EXTRA / source_name, extra / source_name)
    with (extra / name).open("a") as stream:
        stream.write(" ")
    with pytest.raises(ValueError, match="identity differs"):
        growth.verify_growth_inputs(extra, extra / "independent-verification.json")


def test_prepare_output_preserves_fresh_and_resume_phase_guards(tmp_path):
    fresh = tmp_path / "fresh"
    assert growth.prepare_output(fresh, False) == fresh and fresh.is_dir()
    with pytest.raises(FileExistsError):
        growth.prepare_output(fresh, False)
    interrupted = tmp_path / "interrupted"
    interrupted.mkdir()
    (interrupted / "recovery.pt").write_bytes(b"fixture")
    assert growth.prepare_output(interrupted, True) == interrupted
    for forbidden in ("report.json", "failure.json", "model.pt", "dev-predictions.jsonl"):
        path = interrupted / forbidden
        path.write_text("{}")
        with pytest.raises(ValueError, match="interrupted training-phase"):
            growth.prepare_output(interrupted, True)
        path.unlink()


def test_run_delegates_exact_frozen_control_without_mutating_engine(tmp_path):
    qualification = tmp_path / "qualification.json"
    qualification.write_text("{}")
    output = tmp_path / "output"
    calls = []
    def train_fn(*args):
        calls.append(args)
    original_extra = engine.EXTRA
    original_rows = engine.EXPECTED_TRAINING_ROWS
    growth.run(output, qualification, train_fn=train_fn)
    assert engine.EXTRA == original_extra and engine.EXPECTED_TRAINING_ROWS == original_rows
    assert len(calls) == 1
    actual_output, rows, pins, config, actual_qualification, epochs, resume, extra = calls[0]
    assert actual_output == output and len(rows) == 7704
    assert actual_qualification == qualification and epochs == 8 and resume is False
    assert extra == growth.EXTRA and config == engine.SentenceCTCConfig()
    assert pins == growth.growth_pins()
    assert not (output / "failure.json").exists()


def test_run_records_failure_with_growth_pins(tmp_path):
    qualification = tmp_path / "qualification.json"
    qualification.write_text("{}")
    output = tmp_path / "output"
    def fail(*args):
        raise RuntimeError("fixture failure")
    with pytest.raises(RuntimeError, match="fixture failure"):
        growth.run(output, qualification, train_fn=fail)
    report = json.loads((output / "failure.json").read_text())
    assert report["completed"] is False and report["production_changed"] is False
    assert report["pins"] == growth.growth_pins()
