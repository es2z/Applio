"""Fill unvoiced frames of an F0 contour from their voiced neighbours.

Used by the gap-filled methods (f0_methods.GAP_FILLED_METHODS). A frame with F0 = 0
mid-speech switches the generator's excitation from a sine to noise, its pitch
embedding to the unvoiced row and, with protect < 0.5, its content features back to the
unretrieved ones; filling removes all three switches. This is exactly the processing
that was listened to as "B" in docs/f0-benchmarks.md, section 5.

Filling every gap also gives breaths a pitch, which a model trained on unfilled F0 has
never seen there; it sounds them as a tone (docs/f0-benchmarks.md, section 5.8). With
max_gap only the short gaps - consonants - are filled and breaths and pauses stay 0.
"""

import numpy as np


def fill_unvoiced_gaps(f0, max_gap=None):
    """Every frame at or below 0 Hz takes the log-linear interpolation of the voiced
    frames around it; frames before the first / after the last voiced frame take that
    frame's pitch. With fewer than two voiced frames the contour is returned unchanged.

    max_gap: fill only runs of fewer than this many unvoiced frames and leave longer
    ones at 0. A run at either end counts by its own length too, so in realtime the
    newest frames are filled while their gap is still short.
    """
    f0 = np.asarray(f0)
    voiced = np.flatnonzero(f0 > 0)
    if len(voiced) < 2:
        return f0
    filled = np.exp(np.interp(np.arange(len(f0)), voiced, np.log(f0[voiced])))
    filled = filled.astype(f0.dtype)
    filled[voiced] = f0[voiced]  # exact, rather than through exp(log(x))
    if max_gap is None:
        return filled
    out = f0.copy()
    out[voiced] = f0[voiced]
    i, n = 0, len(f0)
    while i < n:
        if f0[i] > 0:
            i += 1
            continue
        j = i
        while j < n and not f0[j] > 0:
            j += 1
        if j - i < max_gap:
            out[i:j] = filled[i:j]
        i = j
    return out
