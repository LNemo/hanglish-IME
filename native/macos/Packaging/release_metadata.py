"""Normalize canonical metadata while rejecting machine-local provenance."""
from __future__ import annotations

from pathlib import PurePosixPath, PureWindowsPath
import re
from urllib.parse import urlsplit

# These are project namespaces, not arbitrary suffixes guessed from a filename.
REPOSITORY_ANCHORS = ('data', 'scripts', 'hanglish', 'checkpoints', '.omx')
PRIVATE_PATH = re.compile(
    r'(?:^|[\s\"\'\(\[\{,:])(?:'
    r'file://|'
    r'/(?:Users|home|private|var(?:/folders)?|tmp|Volumes|opt)(?:/|$)|'
    r'[A-Za-z]:[\\/](?:Users|home|private|var|tmp|Volumes|opt)(?:[\\/]|$)'
    r')',
    re.IGNORECASE,
)
HTTP_URL = re.compile(r'https?://[^\s<>\"\']+', re.IGNORECASE)
EMBEDDED_ABSOLUTE_PATH = re.compile(
    r'(?:^|[\s\"\'\(\[\{,:=])'
    r'(?:/(?!/)[^\s<>\"\']+|[A-Za-z]:[\\/][^\s<>\"\']+)'
)


def _is_http_url(value: str) -> bool:
    parsed = urlsplit(value)
    return (parsed.scheme.lower() in ('http', 'https') and bool(parsed.netloc)
            and not any(character.isspace() for character in value))


def _is_absolute_path(value: str) -> bool:
    return value.startswith('/') or PureWindowsPath(value).is_absolute()


def _repository_relative_path(value: str) -> str:
    normalized = value.replace('\\', '/')
    if re.match(r'^[A-Za-z]:/', normalized):
        normalized = normalized[2:]
    parts = PurePosixPath(normalized).parts
    if '..' in parts or '\n' in value:
        raise ValueError('Unrecognized local path in release metadata')
    for index, part in enumerate(parts):
        if part in REPOSITORY_ANCHORS:
            relative = '/'.join(parts[index:])
            if relative and not PRIVATE_PATH.search(relative):
                return relative
    raise ValueError('Unrecognized local path in release metadata')


def _public_string(value: str) -> str:
    # Source and license URLs are public identifiers even when their URL path
    # happens to contain a local-looking segment such as /home/ or /private/.
    if _is_http_url(value):
        return value
    if _is_absolute_path(value):
        return _repository_relative_path(value)
    # Remove complete web URL tokens before looking for a path embedded in a
    # log message or provenance description. file:// remains private.
    without_urls = HTTP_URL.sub('', value)
    if (not PRIVATE_PATH.search(without_urls)
            and not EMBEDDED_ABSOLUTE_PATH.search(without_urls)):
        return value
    # Avoid echoing developer paths or unknown text into build logs.
    raise ValueError('Unrecognized local path in release metadata')


def sanitize_metadata(value):
    """Return a new JSON value preserving numbers, fields, hashes and attribution.

    Absolute project paths become repository-relative provenance identifiers.
    Unknown private paths and normalization collisions stop packaging for review.
    This does not change model tensors or grant/relabel any third-party license.
    """
    if isinstance(value, str):
        return _public_string(value)
    if isinstance(value, list):
        return [sanitize_metadata(item) for item in value]
    if isinstance(value, dict):
        result = {}
        for key, item in value.items():
            normalized = _public_string(key)
            if normalized in result:
                raise ValueError('Metadata key collision after path normalization')
            result[normalized] = sanitize_metadata(item)
        return result
    return value
