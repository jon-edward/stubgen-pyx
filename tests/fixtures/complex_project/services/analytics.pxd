from libc.stdint cimport uint64_t

cdef class Histogram:
    cdef dict _bins
    cdef readonly int bucket_count
    cpdef void add(self, double value)
    cpdef dict counts(self)

cdef api uint64_t count_values(double[:] values)
