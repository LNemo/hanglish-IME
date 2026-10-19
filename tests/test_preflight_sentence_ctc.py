from scripts.preflight_sentence_ctc import frame_audit, estimate


def row(source, target):
    return dict(id=source, input=source, expected=[target])


def test_frame_audit_counts_repeated_labels_and_never_hides_overflow():
    report = frame_audit([row('a a', '아 아'), row('a b', 'aaaaaaaa')])
    assert report['rows'] == 2
    assert report['invalid'] == 1  # eight labels plus seven separating blanks >9 frames
    assert report['max_required_per_source_byte'] == 5


def test_frame_audit_preserves_ar_length_contract():
    assert frame_audit([row('a ' + 'b' * 509, '아')])['invalid'] == 1
    assert frame_audit([row('a ' + 'b' * 100, '아' * 171)])['invalid'] == 1


def test_estimate_uses_measured_upper_bins_and_noise_envelope():
    rows = [row('a a', '아 아')]
    steps = [dict(source_tokens=128, elapsed_s=2.), dict(source_tokens=256, elapsed_s=1.),
             dict(source_tokens=384, elapsed_s=3.), dict(source_tokens=512, elapsed_s=4.)]
    assert estimate(steps, rows, epochs=2) == 5.
