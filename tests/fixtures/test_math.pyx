"""Mathematical utilities for scientific computing."""

cdef class Matrix:
    """A simple matrix class."""

    cdef int rows
    cdef int cols

    def __init__(self, int rows, int cols):
        """Initialize a matrix."""
        self.rows = rows
        self.cols = cols

    def shape(self) -> tuple[int, int]:
        """Get matrix dimensions."""
        return (self.rows, self.cols)

    cpdef scale(self, double factor):
        """Scale all elements."""
        pass

    cdef int _validate(self):
        """Internal validation (not exposed)."""
        return 0

def matrix_product(Matrix a, Matrix b) -> Matrix:
    """Compute matrix product."""
    return Matrix(a.rows, b.cols)
