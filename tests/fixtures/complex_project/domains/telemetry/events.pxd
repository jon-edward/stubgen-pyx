from libc.stdint cimport uint64_t

cpdef enum EventKind:
    EVENT_START
    EVENT_STOP
    EVENT_ERROR

cdef class Event:
    cdef readonly uint64_t timestamp
    cdef readonly str name
    cdef public EventKind kind
    cpdef str format(self)
