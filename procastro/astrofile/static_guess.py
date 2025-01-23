import re
from pathlib import Path

from procastro.logging import io_logger


def static_guess_spectral_offset(meta) -> dict:
    """The idea is for this function to guess the instrument after reading the meta information."""
    imacs_f2_offset = {4: 0,
                       7: 35 + 2048,
                       3: 0,
                       8: 35 + 2048,
                       2: 0,
                       6: 35 + 2048,
                       1: 0,
                       5: 35 + 2048,
                       }

    # for now, only imacs is supported
    return imacs_f2_offset


def static_guess_type_from_file(filename, hints):
    if hints is None:
        hints = {}

    if isinstance(hints, str):
        hints = {'force': hints}

    if 'force' in hints:
        return hints['force']

    filename = Path(filename).name

    if filename in hints:
        return hints[filename]

    for hint, value in hints.items():
        hint = hint.replace("*", ".*?")
        if re.search(hint, filename):
            return value

    io_logger.warning(f"Assuming image file for '{filename}'")
    return 'img'
