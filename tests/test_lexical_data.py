from hanglish.core.lexical_data import invalid_past_duplicate, read_lexicon


def test_past_contraction_filter_preserves_lexical_nouns():
    def row(surface, parts):
        return [surface, '', '', '1', 'VV+EP', '', '', '', '', '', '', parts]
    decomposition = '반짝이/VV/*+었/EP/*'
    malformed = row('반짝엿', decomposition)
    assert invalid_past_duplicate(malformed, {('반짝였', decomposition)})
    assert not invalid_past_duplicate(malformed, set())
    assert not invalid_past_duplicate(row('반짝였', decomposition), {('반짝였', decomposition)})
    assert not invalid_past_duplicate(row('엿', '엿/NNG/*'), {('였', '엿/NNG/*')})


def test_snapshot_filter_is_morphology_scoped():
    from pathlib import Path
    path = Path(__file__).resolve().parents[1] / 'data/external/mecab-ko-dic'
    entries, fingerprint, rejected = read_lexicon(path)
    assert rejected == 2
    surfaces = {word for word, _, _ in entries}
    assert '번쩍엿' not in surfaces
    assert {'번쩍였', '엿', '였'} <= surfaces
    assert len(fingerprint) == 64


def test_ni_evidence_requires_predicate_and_connective_analysis(tmp_path):
    from hanglish.core.lexical_data import read_attested_ni_forms
    path = tmp_path / 'Inflect.csv'
    path.write_text('들으니,0,0,1,VV+EC,*,F,들으니,Inflect,VV,EC,듣/VV/*+으니/EC/*\n'
                    '먹니,0,0,1,VV+EF,*,F,먹니,Inflect,VV,EF,먹/VV/*+니/EF/*\n'
                    '책이니,0,0,1,NNG+EC,*,F,책이니,Inflect,NNG,EC,책/NNG/*+니/EC/*\n')
    assert read_attested_ni_forms(tmp_path) == {'듣': {'들으니'}}


def test_counter_grammar_reads_only_hangul_nnbc_without_expanding_word_lexicon(tmp_path):
    from hanglish.core.lexical_data import read_counter_units
    assert read_counter_units(tmp_path) == ()
    path = tmp_path / 'NNBC.csv'
    path.write_text('주,0,0,100,NNBC,*,F,주,*,*,*,*\n'
                    '주일,0,0,200,NNBC,*,T,주일,*,*,*,*\n'
                    'kg,0,0,200,NNBC,*,F,kg,*,*,*,*\n'
                    '사과,0,0,200,NNG,*,F,사과,*,*,*,*\n')
    assert read_counter_units(tmp_path) == ('주일', '주')
    assert read_lexicon(tmp_path)[0] == []
