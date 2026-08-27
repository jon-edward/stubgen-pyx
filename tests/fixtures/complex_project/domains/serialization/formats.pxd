from libc.stdint cimport int32_t, uint32_t

ctypedef struct Header:
    uint32_t version
    uint32_t length
    int32_t flags

ctypedef union Payload:
    int32_t integer
    double decimal
    const char* text

cdef extern from *:
    ctypedef struct Packet:
        Header header
        Payload payload
        void (*release)(void*)
