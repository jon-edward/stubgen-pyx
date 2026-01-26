"""
This is a test of importing modules.
"""

from dataclasses import dataclass

from typing import List, Dict, Tuple as tup
import typing

cpdef int imported_func():
    return 1


@dataclass
class RegularDataclass:
    x: int
    y: float
    z: complex


def returns_tuple() -> tup[typing.Any, int]:
    return 1, 2


# def returns_list() -> List[int]:
#     return [1, 2]


def returns_dict() -> Dict[int, int]:
    return {1: 2}


cdef class ImportedClass:
    """
    A docstring for ImportedClass
    """

    def __init__(self):
        """
        A docstring for __init__
        """
        self._private_field = 1
    
    cpdef int public_method(self):
        return self._private_field
    
    @property
    def public_field(self) -> int:
        return self._private_field

    @public_field.setter
    def public_field(self, value: int):
        self._private_field = value


X: ImportedClass = ImportedClass()