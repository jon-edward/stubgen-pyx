from libc.stdint cimport uint32_t

ctypedef fused WireValue:
    int
    double
    str

cdef api uint32_t checksum(bytes payload)
