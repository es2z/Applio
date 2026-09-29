"""Fill unvoiced frames of an F0 contour from their voiced neighbours.

Used by the gap-filled methods (f0_methods.GAP_FILLED_METHODS). A frame with F0 = 0
mid-speech switches the generator's excitation from a sine to noise, its pitch
embedding to the unvoiced row and, with protect < 0.5, its content features back to the
unretrieved ones; filling removes all three switches. This is exactly the processing
that was listened to as "B" in docs/f0-benchmarks.md, section 5.
"""

import numpy as np


def fill_unvoiced_gaps(f0):
    """Every frame at or below 0 Hz takes the log-linear interpolation of the voiced
    frames around it; frames before the first / after the last voiced frame take that
    frame's pitch. With fewer than two voiced frames the contour is returned unchanged.
    """
    f0 = np.asarray(f0)
    voiced = np.flatnonzero(f0 > 0)
    if len(voiced) < 2:
        return f0
    filled = np.exp(np.interp(np.arange(len(f0)), voiced, np.log(f0[voiced])))
    filled = filled.astype(f0.dtype)
    filled[voiced] = f0[voiced]  # exact, rather than through exp(log(x))
    return filled
