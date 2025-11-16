cpdef int imported_func()

cdef class ImportedClass:
    cdef public int _private_field

    cpdef int public_method(self)
