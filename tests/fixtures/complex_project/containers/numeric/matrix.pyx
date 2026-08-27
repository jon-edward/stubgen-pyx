"""A small C++-backed matrix extension type."""

from __future__ import annotations

from typing import Iterable, Iterator, Sequence

from libcpp.vector cimport vector

include "matrix.pxi"


cdef class Matrix:
    cdef vector[double] _values
    cdef readonly Py_ssize_t rows
    cdef readonly Py_ssize_t columns

    def __init__(self, rows: int, columns: int, values: Iterable[float] = ()) -> None:
        if rows < 0 or columns < 0 or rows * columns > MAX_MATRIX_SIZE:
            raise ValueError("matrix dimensions are out of range")
        self.rows = rows
        self.columns = columns
        self._values.assign(rows * columns, 0.0)
        for index, value in enumerate(values):
            if index >= self._values.size():
                break
            self._values[index] = value

    cpdef double item(self, Py_ssize_t row, Py_ssize_t column):
        if not 0 <= row < self.rows or not 0 <= column < self.columns:
            raise IndexError("matrix index out of range")
        return self._values[flat_index(row, column, self.columns)]

    def row(self, row: int) -> tuple[float, ...]:
        return tuple(self.item(row, column) for column in range(self.columns))

    def __iter__(self) -> Iterator[tuple[float, ...]]:
        return (self.row(row) for row in range(self.rows))
