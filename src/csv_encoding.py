'''Resolve CSV encodings with deterministic, strict full-file decoding.'''

import codecs
from collections.abc import Iterable
from pathlib import Path

PathLike = str | Path

DEFAULT_CSV_ENCODINGS = (
    'utf-8-sig',
    'utf-8',
    'gb18030',
    'gbk',
    'gb2312',
    'big5',
    'cp1252',
    'utf-16',
    'utf-16-le',
    'utf-16-be',
    'latin-1',
)

_BOM_ENCODINGS = (
    (codecs.BOM_UTF32_LE, 'utf-32'),
    (codecs.BOM_UTF32_BE, 'utf-32'),
    (codecs.BOM_UTF8, 'utf-8-sig'),
    (codecs.BOM_UTF16_LE, 'utf-16'),
    (codecs.BOM_UTF16_BE, 'utf-16'),
)


def _canonical_encoding(encoding: str) -> str:
    return codecs.lookup(encoding).name


def _encoding_candidates(
    filepath: Path,
    preferred_encoding: str | None,
    encodings: Iterable[str],
) -> list[str]:
    if preferred_encoding is not None:
        return [_canonical_encoding(preferred_encoding)]

    with filepath.open('rb') as stream:
        prefix = stream.read(4)

    names = []
    for bom, encoding in _BOM_ENCODINGS:
        if prefix.startswith(bom):
            names.append(encoding)
            break
    names.extend(encodings)

    candidates = []
    seen = set()
    for encoding in names:
        canonical = _canonical_encoding(encoding)
        if canonical not in seen:
            seen.add(canonical)
            candidates.append(canonical)

    return candidates


def _decode_file(filepath: Path, encoding: str, chunk_size: int) -> None:
    decoder = codecs.getincrementaldecoder(encoding)(errors='strict')
    with filepath.open('rb') as stream:
        while chunk := stream.read(chunk_size):
            decoder.decode(chunk, final=False)
    decoder.decode(b'', final=True)


def detect_csv_encoding(
    filepath: PathLike,
    preferred_encoding: str | None = None,
    encodings: Iterable[str] = DEFAULT_CSV_ENCODINGS,
    chunk_size: int = 1024 * 1024,
) -> str:
    '''Return the first candidate that strictly decodes the complete file.'''
    path = Path(filepath)
    candidates = _encoding_candidates(path, preferred_encoding, encodings)
    last_error = None

    for encoding in candidates:
        try:
            _decode_file(path, encoding, chunk_size)
            return encoding
        except UnicodeError as error:
            last_error = error

    tried = ', '.join(candidates)
    raise UnicodeError(
        f'Unable to decode CSV file {path} with candidate encodings: {tried}'
    ) from last_error
