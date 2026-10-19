"""Experiment: keep bounded forward-valid mechanical alternatives to neural OOV output."""
from hanglish.core.phonology import romanize_word


def preserved_options(converter, source, learned, limit=4):
    if type(limit) is not int or not 1 <= limit <= 8:
        raise ValueError('fallback limit must be an integer from 1 to 8')
    baseline = converter._word_options(source, learned)
    if (not learned or not any(o.kind == 'neural' for o in baseline)
            or any(o.score >= 5.5 for o in baseline)):
        return baseline
    seen = {o.text for o in baseline}
    added = []
    for option in converter.generator.options(source):
        if (option.kind != 'syllable' or option.text in seen
                or romanize_word(option.text) != source):
            continue
        added.append(option)
        seen.add(option.text)
        if len(added) == limit:
            break
    return baseline + tuple(added)
