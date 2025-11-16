"""
This is a module for testing stub file generation.
"""

from typing import List

# Hides imports that are not used in the stub
import pprint
from dataclasses import dataclass

from .test2 cimport imported_func, ImportedClass

__cimport_types__ = {
    ImportedClass,
}

@dataclass
class PythonClass:
    """
    Injects type annotations when possible.
    """
    x: int
    y: int
    z: int


def func() -> ImportedClass:
    pass


cdef class TestClass2:
    def __init__(self):
        """
        A docstring for __init__
        """


cdef class TestClass:
    """
    This is a class for testing stub file generation.
    """

    a: int
    cdef public TestClass2 _test_class_2

    def __init__(self):
        """
        A docstring for __init__
        """
        self.a = imported_func()
        self._test_class_2 = TestClass2()
    
    def test_class_2(self) -> TestClass2:
        return self._test_class_2
    
    cpdef b(self):
        """
        A docstring for b
        """
        return self.a
    
    def c(self):
        """
        A docstring for c
        """
        return self.a
    
    cdef d(self):
        """
        A docstring for d (this should be ignored)
        """
        return self.a

cdef bint private_function(int x):
    """
    Hides cdef functions from the stub.
    """
    return False


cpdef bint public_function(int x):
    """
    Shows cpdef functions in the stub.
    """
    return False

cpdef ImportedClass imported_class_test():
    return ImportedClass()


def print_dict(d: dict) -> List[bint]:
    """
    Follows imports.
    """  
    pprint.pprint(d)
    return [False, False, True]
