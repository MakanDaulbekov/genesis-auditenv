"""Пик резидентной памяти (RSS) текущего процесса в МБ — для диагностики в REPORT.md.

Linux/macOS: resource.getrusage (ru_maxrss: КБ на Linux, байты на macOS).
Windows: PeakWorkingSetSize через psapi.GetProcessMemoryInfo (ctypes, без зависимостей).
Если замер недоступен — None (отчёт печатает «—»).
"""
from __future__ import annotations

import sys


def self_peak_rss_mb() -> float | None:
    try:
        import resource
    except ImportError:
        return _windows_peak_mb()
    v = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return round(v / (2**20 if sys.platform == "darwin" else 1024), 1)


def children_peak_rss_mb() -> float | None:
    """Пик среди завершённых дочерних процессов (только POSIX); на Windows — None."""
    try:
        import resource
    except ImportError:
        return None
    v = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss
    return round(v / (2**20 if sys.platform == "darwin" else 1024), 1)


def _windows_peak_mb() -> float | None:
    try:
        import ctypes
        from ctypes import wintypes

        class _PMC(ctypes.Structure):
            _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD),
                        ("PeakWorkingSetSize", ctypes.c_size_t), ("WorkingSetSize", ctypes.c_size_t),
                        ("QuotaPeakPagedPoolUsage", ctypes.c_size_t), ("QuotaPagedPoolUsage", ctypes.c_size_t),
                        ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t), ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                        ("PagefileUsage", ctypes.c_size_t), ("PeakPagefileUsage", ctypes.c_size_t)]

        k32 = ctypes.WinDLL("kernel32")
        psapi = ctypes.WinDLL("psapi")
        k32.GetCurrentProcess.restype = wintypes.HANDLE
        psapi.GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.POINTER(_PMC), wintypes.DWORD]
        psapi.GetProcessMemoryInfo.restype = wintypes.BOOL
        pmc = _PMC()
        pmc.cb = ctypes.sizeof(_PMC)
        if not psapi.GetProcessMemoryInfo(k32.GetCurrentProcess(), ctypes.byref(pmc), pmc.cb):
            return None
        return round(pmc.PeakWorkingSetSize / 2**20, 1)
    except Exception:
        return None
