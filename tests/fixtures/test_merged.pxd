cdef class MergedClass:
    cdef public int merged_value
    cpdef int function(self, int x = *)
