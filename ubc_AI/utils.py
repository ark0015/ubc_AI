import numpy as np


def fromfile(file, dtype, count, *args, **kwargs):
    """Wrapper around np.fromfile to support any file-like object"""

    #    try:
    #        return np.fromfile(file, dtype=dtype, count=count, *args, **kwargs)
    #    except (TypeError, IOError, UnsupportedOperation):
    #        return np.frombuffer(
    #            file.read(int(count * np.dtype(dtype).itemsize)),
    #            dtype=dtype, count=count, *args, **kwargs)
    #    """Read from any file-like object into a numpy array"""

    itemsize = np.dtype(dtype).itemsize
    buffer = np.zeros(count * itemsize, np.uint8)
    bytes_read = -1
    offset = 0
    while bytes_read != 0:
        bytes_read = file.readinto(buffer[offset:])
        offset += bytes_read
    rounded_bytes = (offset // itemsize) * itemsize
    buffer = buffer[:rounded_bytes]
    buffer.dtype = dtype
    return buffer
