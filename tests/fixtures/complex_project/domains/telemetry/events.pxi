from libc.stdint cimport uint64_t

cdef inline uint64_t timestamp_delta(uint64_t first, uint64_t second):
    return second - first if second >= first else 0
