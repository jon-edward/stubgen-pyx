DEF CURRENT_VERSION = 4

cdef inline int checked_length(bytes value):
    if len(value) > 0xFFFFFFFF:
        raise OverflowError("payload is too large")
    return len(value)
