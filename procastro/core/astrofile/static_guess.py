

def static_guess_spectral_offset(meta) -> dict:
    """"The idea is for this function to guess the instrument after reading the meta information."""
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
