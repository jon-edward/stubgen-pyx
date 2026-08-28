from libcpp.vector cimport vector

cdef class Matrix:
    cdef vector[double] _values
    cdef readonly Py_ssize_t rows
    cdef readonly Py_ssize_t columns
    cpdef double item(self, Py_ssize_t row, Py_ssize_t column)
