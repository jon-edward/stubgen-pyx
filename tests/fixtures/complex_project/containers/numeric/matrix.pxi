DEF MAX_MATRIX_SIZE = 4096

cdef inline Py_ssize_t flat_index(Py_ssize_t row, Py_ssize_t column, Py_ssize_t columns):
    return row * columns + column
