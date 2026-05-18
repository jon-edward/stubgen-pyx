cdef class MergedClass:
    original_value: int = 0
    python_attribute: int = 1

    def __init__(self, int v):
        self.original_value = v
        self.merged_value = v

    cpdef int function(self):
        return self.merged_value

    def set_merged_value(self, int v):
        self.merged_value = v
