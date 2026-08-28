DEF WIRE_VERSION = 3

def _versioned(payload: bytes) -> bytes:
    return bytes((WIRE_VERSION,)) + payload
