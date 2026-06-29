cpdef int imported_func()

cdef class ImportedClass:
    cdef public int _private_field

    cdef public double[:, :] embedding

    cpdef int public_method(self)
