DEF DEFAULT_BUCKETS = 10

cdef inline int bucket_for(double value, double width):
    return <int>(value / width)
