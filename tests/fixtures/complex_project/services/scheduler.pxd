from libc.stdint cimport uint64_t

ctypedef enum JobState:
    JOB_PENDING
    JOB_RUNNING
    JOB_COMPLETE
    JOB_FAILED

cdef class Job:
    cdef readonly uint64_t identifier
    cdef public JobState state
    cpdef bint ready(self, uint64_t now)
