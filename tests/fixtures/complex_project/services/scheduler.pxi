from libc.stdint cimport uint64_t

cdef inline bint is_due(uint64_t scheduled_at, uint64_t now):
    return now >= scheduled_at
