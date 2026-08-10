from __future__ import annotations

import xml.etree.ElementTree   # dotted import; binds `xml` in scope

cdef class MyClass:
    @property
    def handle(self) -> xml.etree.ElementTree.Element:
        """Return an Element."""
        pass
