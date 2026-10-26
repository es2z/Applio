def strtobool(value):
    """distutils.util.strtobool, which went away with distutils in Python 3.12.

    Returns 1 for y, yes, t, true, on and 1, 0 for n, no, f, false, off and 0, case
    insensitively, and raises ValueError for anything else, exactly as distutils did.
    """
    value = value.lower()
    if value in ("y", "yes", "t", "true", "on", "1"):
        return 1
    if value in ("n", "no", "f", "false", "off", "0"):
        return 0
    raise ValueError(f"invalid truth value {value!r}")
