"""DirectShow plumbing: device names, and the two camera-control interfaces.

IAMVideoProcAmp (brightness, contrast, white balance, anti-flicker ...) and
IAMCameraControl (exposure, focus, zoom ...) are what Logi Tune and the Windows
camera property page use. Neither is reachable through OpenCV, so they are
declared here directly with comtypes; pygrabber already provides the device
enumeration that binds a camera's capture filter by index, in the same order
OpenCV's DirectShow backend numbers devices.

Every COM call made through this module runs on one dedicated thread. A COM
object belongs to the thread that created it, and FastAPI answers requests from
a pool of threads -- calling the same interface from whichever thread happens to
serve a request is undefined behaviour, so `run()` hands each call to the one
thread that owns them all.
"""

from __future__ import annotations

import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable, TypeVar

T = TypeVar("T")

AVAILABLE = False
if sys.platform == "win32":
    try:
        from ctypes import HRESULT, POINTER, c_long

        import comtypes
        from comtypes import COMMETHOD, GUID, IUnknown
        from pygrabber.dshow_graph import SystemDeviceEnum
        from pygrabber.dshow_ids import DeviceCategories

        AVAILABLE = True
    except ImportError:
        pass

# Flags carried by every Set/Get. The two interfaces use the same values.
FLAG_AUTO = 0x1
FLAG_MANUAL = 0x2

# Generous: a driver that is merely slow is not an error, a hung one is.
CALL_TIMEOUT_S = 10.0


if AVAILABLE:
    def _control_methods() -> list:
        # Both interfaces share this exact vtable: GetRange, Set, Get (strmif.h).
        return [
            COMMETHOD([], HRESULT, "GetRange",
                      (["in"], c_long, "Property"),
                      (["out"], POINTER(c_long), "pMin"),
                      (["out"], POINTER(c_long), "pMax"),
                      (["out"], POINTER(c_long), "pSteppingDelta"),
                      (["out"], POINTER(c_long), "pDefault"),
                      (["out"], POINTER(c_long), "pCapsFlags")),
            COMMETHOD([], HRESULT, "Set",
                      (["in"], c_long, "Property"),
                      (["in"], c_long, "lValue"),
                      (["in"], c_long, "Flags")),
            COMMETHOD([], HRESULT, "Get",
                      (["in"], c_long, "Property"),
                      (["out"], POINTER(c_long), "lValue"),
                      (["out"], POINTER(c_long), "Flags")),
        ]

    class IAMVideoProcAmp(IUnknown):
        _iid_ = GUID("{C6E13360-30AC-11d0-A18C-00A0C9118956}")
        _methods_ = _control_methods()

    class IAMCameraControl(IUnknown):
        _iid_ = GUID("{C6E13370-30AC-11d0-A18C-00A0C9118956}")
        _methods_ = _control_methods()


# ---------------------------------------------------------------- COM thread

_pool: ThreadPoolExecutor | None = None
_pool_lock = threading.Lock()


def _init_thread() -> None:
    comtypes.CoInitialize()


def run(fn: Callable[..., T], *args: Any) -> T:
    """Run `fn(*args)` on the COM thread and return its result."""
    if not AVAILABLE:
        raise RuntimeError("DirectShow camera controls need Windows with pygrabber installed.")
    global _pool
    with _pool_lock:
        if _pool is None:
            _pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="dshow",
                                       initializer=_init_thread)
    return _pool.submit(fn, *args).result(timeout=CALL_TIMEOUT_S)


# ---------------------------------------------------------------- devices

def device_names() -> list[str]:
    """Friendly names of every video input, in DirectShow (= OpenCV) order."""
    if not AVAILABLE:
        return []

    def _list() -> list[str]:
        return list(SystemDeviceEnum().get_available_filters(
            DeviceCategories.VideoInputDevice))

    try:
        return run(_list)
    except Exception:
        # pygrabber raises assorted COM errors when no device is present.
        return []


def bind(index: int) -> tuple[Any, Any]:
    """(IAMVideoProcAmp, IAMCameraControl) for device `index`; either may be None.

    Must be called on the COM thread (through `run`). This binds a second
    instance of the device's capture filter, used only for control: the kernel
    driver serves property requests from any open handle, so it works while
    OpenCV is streaming from the same camera.
    """
    filt, _name = SystemDeviceEnum().get_filter_by_index(
        DeviceCategories.VideoInputDevice, index)
    return _query(filt, IAMVideoProcAmp), _query(filt, IAMCameraControl)


def _query(filt: Any, interface: Any) -> Any:
    try:
        return filt.QueryInterface(interface)
    except comtypes.COMError:
        return None
