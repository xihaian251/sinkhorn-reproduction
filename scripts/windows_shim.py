"""Import-time compatibility shim for residual-sinkhorn 0.2.0 on Windows.

The original 0.2.0 wheel calls ``ctypes.CDLL(None).memcmp`` at import time.
On Windows/CPython that raises TypeError, which the author's handler
``except (AttributeError, OSError)`` does not catch.  The shim makes CDLL(None)
raise OSError so the author's OWN ``np.array_equal`` fallback is selected.

This does not modify the wheel or the installed source and does not change the
solver path.  It only affects the warm-cache cost comparison (bytewise memcmp
vs. np.array_equal).  Author released 0.2.1 with a native fix; see README.
"""
import ctypes


def install():
    original = ctypes.CDLL

    def _portable_cdll(name, *args, **kwargs):
        if name is None:
            raise OSError("CDLL(None) unsupported on this platform; use numpy fallback")
        return original(name, *args, **kwargs)

    ctypes.CDLL = _portable_cdll
    try:
        import residual_sinkhorn as rs
    finally:
        ctypes.CDLL = original
    return rs
