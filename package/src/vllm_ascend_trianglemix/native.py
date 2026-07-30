#
# Copyright 2026 TriangleMix contributors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
"""Lazy bootstrap for the wheel-bundled TriangleMix native operator.

Importing this module never modifies the process environment and never loads
a shared object.  :func:`ensure_native_loaded` performs those actions only
when its ``enabled`` argument is true.  A non-strict failure is represented by
an immutable, queryable status so the attention backend can safely use FIA.
"""

from __future__ import annotations

import ctypes
import importlib
import os
import platform
import re
import sys
import threading
from dataclasses import asdict, dataclass, replace
from importlib import metadata as importlib_metadata
from pathlib import Path
from types import ModuleType
from typing import Any


ADAPTER_FILENAME = (
    "triangle_paged_attention_torch.cpython-310-aarch64-linux-gnu.so"
)
EXPECTED_SYSTEM = "Linux"
EXPECTED_MACHINE = "aarch64"
EXPECTED_PYTHON_IMPLEMENTATION = "cpython"
EXPECTED_PYTHON_VERSION = (3, 10)
EXPECTED_SOC_FAMILY = "Ascend 910B"
EXPECTED_CANN_COMPILER = "9.0.1"
SUPPORTED_RUNTIME_FINGERPRINTS = (
    (
        "0.23.0",
        "0.23.0rc1",
        "2.10.0",
        "2.10.0.post2",
    ),
)
_RUNTIME_DISTRIBUTIONS = (
    ("vllm", "vllm", "vLLM"),
    ("vllm_ascend", "vllm-ascend", "vLLM-Ascend"),
    ("torch", "torch", "PyTorch"),
    ("torch_npu", "torch-npu", "torch_npu"),
)

_LOAD_LOCK = threading.RLock()
_CUST_OPAPI_HANDLE: ctypes.CDLL | None = None


@dataclass(frozen=True)
class NativePaths:
    """Absolute paths to immutable artifacts installed inside this wheel."""

    adapter: str
    opp_root: str
    opp_vendor: str
    op_api_library_dir: str
    cust_opapi: str


@dataclass(frozen=True)
class NativeStatus:
    """Compatibility and load state for the current worker process."""

    enabled: bool
    loaded: bool
    compatible: bool
    state: str
    system: str
    machine: str
    python_implementation: str
    python_version: str
    vllm_version: str | None = None
    vllm_ascend_version: str | None = None
    soc_version: str | None = None
    torch_version: str | None = None
    torch_npu_version: str | None = None
    torch_distribution_version: str | None = None
    torch_npu_distribution_version: str | None = None
    runtime_fingerprint: tuple[str, str, str, str] | None = None
    cann_compiler_version: str | None = None
    adapter_path: str | None = None
    opp_root: str | None = None
    opp_vendor_path: str | None = None
    cust_opapi_path: str | None = None
    errors: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()

    def __bool__(self) -> bool:
        """Treat a status as true only after operator registration succeeds."""
        return self.loaded

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable diagnostic record."""
        value = asdict(self)
        if self.runtime_fingerprint is not None:
            value["runtime_fingerprint"] = list(self.runtime_fingerprint)
        value["errors"] = list(self.errors)
        value["warnings"] = list(self.warnings)
        return value


class NativeBootstrapError(RuntimeError):
    """Raised when strict native bootstrap cannot produce a usable operator."""

    def __init__(self, status: NativeStatus):
        self.status = status
        details = "; ".join(status.errors) or "unknown native bootstrap error"
        super().__init__(f"TriangleMix native bootstrap failed: {details}")


def _runtime_identity() -> tuple[str, str, str, str]:
    return (
        platform.system(),
        platform.machine().lower(),
        sys.implementation.name.lower(),
        f"{sys.version_info.major}.{sys.version_info.minor}."
        f"{sys.version_info.micro}",
    )


def _initial_status() -> NativeStatus:
    system, machine, implementation, python_version = _runtime_identity()
    return NativeStatus(
        enabled=False,
        loaded=False,
        compatible=True,
        state="not_initialized",
        system=system,
        machine=machine,
        python_implementation=implementation,
        python_version=python_version,
    )


_LAST_STATUS = _initial_status()


def native_status() -> NativeStatus:
    """Return the last status without probing resources or changing state."""
    with _LOAD_LOCK:
        return _LAST_STATUS


def _resolve_bundled_paths() -> NativePaths:
    """Resolve package data only for an explicitly enabled feature."""
    native_root = Path(__file__).resolve().parent / "_native"
    opp_root = native_root / "opp"
    vendor = opp_root / "vendors" / "trianglemix"
    op_api_library_dir = vendor / "op_api" / "lib"
    return NativePaths(
        adapter=str((native_root / ADAPTER_FILENAME).resolve()),
        opp_root=str(opp_root.resolve()),
        opp_vendor=str(vendor.resolve()),
        op_api_library_dir=str(op_api_library_dir.resolve()),
        cust_opapi=str((op_api_library_dir / "libcust_opapi.so").resolve()),
    )


def _artifact_errors(paths: NativePaths) -> list[str]:
    errors: list[str] = []
    required_files = {
        "Torch adapter": paths.adapter,
        "custom op API library": paths.cust_opapi,
        "OPP version metadata": os.path.join(
            paths.opp_vendor, "version.info"
        ),
    }
    required_directories = {
        "OPP root": paths.opp_root,
        "OPP vendor": paths.opp_vendor,
    }
    for description, path in required_files.items():
        if not os.path.isfile(path):
            errors.append(
                f"bundled {description} is missing from the wheel: {path}"
            )
    for description, path in required_directories.items():
        if not os.path.isdir(path):
            errors.append(
                f"bundled {description} directory is missing: {path}"
            )
    return errors


def _read_cann_compiler_version(paths: NativePaths) -> str | None:
    version_path = Path(paths.opp_vendor) / "version.info"
    try:
        contents = version_path.read_text(encoding="utf-8")
    except OSError:
        return None
    match = re.search(
        r"custom_opp_compiler_version\s*=\s*([^\s]+)", contents
    )
    return match.group(1) if match is not None else None


def _import_dependency(name: str) -> ModuleType:
    return importlib.import_module(name)


def _distribution_versions(
    errors: list[str],
) -> dict[str, str | None]:
    """Read installed distribution metadata without importing vLLM."""
    versions: dict[str, str | None] = {}
    for key, distribution, description in _RUNTIME_DISTRIBUTIONS:
        try:
            value = str(importlib_metadata.version(distribution)).strip()
        except importlib_metadata.PackageNotFoundError:
            value = None
            errors.append(
                f"{description} distribution metadata is missing "
                f"({distribution!r} is not installed)"
            )
        except Exception as exc:
            value = None
            errors.append(
                f"{description} distribution version cannot be read "
                f"({type(exc).__name__}: {exc})"
            )
        if value is not None and (
            not value or value.casefold() == "unknown"
        ):
            errors.append(
                f"{description} distribution reports an unknown version"
            )
            value = None
        versions[key] = value
    return versions


def _public_version(value: str | None) -> str | None:
    """Return a PEP 440 public version, retaining pre/post/dev markers.

    Local build labels such as ``+cpu`` and ``+empty`` are intentionally
    ignored.  Pre-releases and post-releases remain part of ``public`` and
    therefore cannot accidentally satisfy a final-release allowlist entry.
    The import is lazy so feature-off never imports packaging.
    """
    if value is None:
        return None
    candidate = str(value).strip()
    if not candidate or candidate.casefold() == "unknown":
        return None
    try:
        from packaging.version import Version
    except ModuleNotFoundError:
        try:
            # setuptools vendors the same packaging implementation in
            # minimal build/test environments. Runtime wheels declare the
            # standalone dependency, but this fallback keeps source tests
            # hermetic.
            from setuptools._vendor.packaging.version import (  # type: ignore
                Version,
            )
        except Exception:
            return None
    try:
        return Version(candidate).public
    except Exception:
        return None


def _format_fingerprint(
    fingerprint: tuple[str | None, str | None, str | None, str | None],
) -> str:
    names = ("vllm", "vllm-ascend", "torch", "torch_npu")
    return "(" + ", ".join(
        f"{name}={value or 'missing/unknown'}"
        for name, value in zip(names, fingerprint)
    ) + ")"


def _runtime_fingerprint(
    *,
    vllm_version: str | None,
    vllm_ascend_version: str | None,
    torch_version: str | None,
    torch_npu_version: str | None,
    torch_distribution_version: str | None,
    torch_npu_distribution_version: str | None,
    errors: list[str],
) -> tuple[str, str, str, str] | None:
    """Validate one complete, non-mixable runtime compatibility tuple."""
    raw = (
        vllm_version,
        vllm_ascend_version,
        torch_version,
        torch_npu_version,
    )
    normalized = tuple(_public_version(value) for value in raw)
    descriptions = ("vLLM", "vLLM-Ascend", "PyTorch", "torch_npu")
    for description, value, public in zip(
        descriptions, raw, normalized
    ):
        if value is not None and public is None:
            errors.append(
                f"{description} version {value!r} is unknown or not valid "
                "PEP 440"
            )

    module_and_distribution = (
        (
            "PyTorch",
            torch_version,
            torch_distribution_version,
        ),
        (
            "torch_npu",
            torch_npu_version,
            torch_npu_distribution_version,
        ),
    )
    for description, module_value, distribution_value in (
        module_and_distribution
    ):
        module_public = _public_version(module_value)
        distribution_public = _public_version(distribution_value)
        if (
            module_public is not None
            and distribution_public is not None
            and module_public != distribution_public
        ):
            errors.append(
                f"{description} module/distribution version mismatch: "
                f"module={module_value!r}, "
                f"distribution={distribution_value!r}"
            )

    candidate = (
        normalized[0],
        normalized[1],
        normalized[2],
        normalized[3],
    )
    supported = " or ".join(
        _format_fingerprint(item)
        for item in SUPPORTED_RUNTIME_FINGERPRINTS
    )
    if (
        any(value is None for value in candidate)
        or candidate not in SUPPORTED_RUNTIME_FINGERPRINTS
    ):
        detected = _format_fingerprint(raw)
        errors.append(
            "unsupported complete runtime fingerprint: "
            f"detected {detected}; supported {supported}. "
            "PEP 440 local suffixes (+...) are ignored, but pre-release, "
            "post-release, and dev markers must match exactly"
        )
        return None
    return (
        str(candidate[0]),
        str(candidate[1]),
        str(candidate[2]),
        str(candidate[3]),
    )


def _dependency_versions(
    errors: list[str],
) -> tuple[ModuleType | None, ModuleType | None, str | None, str | None]:
    torch_module: ModuleType | None = None
    torch_npu_module: ModuleType | None = None
    torch_version: str | None = None
    torch_npu_version: str | None = None
    try:
        torch_module = _import_dependency("torch")
        torch_version = str(getattr(torch_module, "__version__", "unknown"))
    except Exception as exc:
        errors.append(
            "PyTorch cannot be imported; install the vLLM-Ascend runtime "
            f"before enabling TriangleMix ({type(exc).__name__}: {exc})"
        )
    try:
        torch_npu_module = _import_dependency("torch_npu")
        torch_npu_version = str(
            getattr(torch_npu_module, "__version__", "unknown")
        )
    except Exception as exc:
        errors.append(
            "torch_npu cannot be imported; install a CANN-compatible "
            f"torch_npu build ({type(exc).__name__}: {exc})"
        )
    if torch_module is not None:
        ops = getattr(torch_module, "ops", None)
        if ops is None or not callable(getattr(ops, "load_library", None)):
            errors.append("PyTorch does not expose torch.ops.load_library")
    return (
        torch_module,
        torch_npu_module,
        torch_version,
        torch_npu_version,
    )


def _environment_soc_version() -> str | None:
    for name in (
        "ASCEND_SOC_VERSION",
        "ASCEND_CHIP_TYPE",
        "ASCEND_COMPUTE_UNIT",
    ):
        value = os.getenv(name)
        if value and value.strip():
            return value.strip()
    return None


def _device_soc_version(torch_npu_module: ModuleType | None) -> str | None:
    """Best-effort SoC lookup after torch_npu is already imported."""
    if torch_npu_module is None:
        return _environment_soc_version()
    npu = getattr(torch_npu_module, "npu", None)
    get_device_name = getattr(npu, "get_device_name", None)
    if not callable(get_device_name):
        return _environment_soc_version()
    try:
        value = get_device_name()
    except TypeError:
        current_device = getattr(npu, "current_device", None)
        if not callable(current_device):
            return _environment_soc_version()
        try:
            value = get_device_name(current_device())
        except Exception:
            return _environment_soc_version()
    except Exception:
        return _environment_soc_version()
    return str(value).strip() or _environment_soc_version()


def _soc_is_supported(soc_version: str) -> bool:
    compact = re.sub(r"[^a-z0-9]", "", soc_version.lower())
    return "910b" in compact


def inspect_native_compatibility(enabled: bool = True) -> NativeStatus:
    """Probe the enabled native runtime without modifying paths or loading it.

    When ``enabled`` is false this function deliberately does not resolve
    package resources and does not import PyTorch or torch_npu.
    """
    system, machine, implementation, python_version = _runtime_identity()
    if not enabled:
        return NativeStatus(
            enabled=False,
            loaded=False,
            compatible=True,
            state="disabled",
            system=system,
            machine=machine,
            python_implementation=implementation,
            python_version=python_version,
        )

    errors: list[str] = []
    warnings: list[str] = []
    if system != EXPECTED_SYSTEM:
        errors.append(
            f"unsupported operating system {system!r}; expected Linux"
        )
    if machine != EXPECTED_MACHINE:
        errors.append(
            f"unsupported machine {machine!r}; expected aarch64"
        )
    if implementation != EXPECTED_PYTHON_IMPLEMENTATION:
        errors.append(
            f"unsupported Python implementation {implementation!r}; "
            "expected CPython"
        )
    if sys.version_info[:2] != EXPECTED_PYTHON_VERSION:
        errors.append(
            f"unsupported Python {python_version}; the bundled adapter "
            "requires CPython 3.10"
        )

    paths = _resolve_bundled_paths()
    errors.extend(_artifact_errors(paths))
    cann_version = _read_cann_compiler_version(paths)
    if (
        cann_version is not None
        and cann_version != EXPECTED_CANN_COMPILER
    ):
        errors.append(
            f"bundled OPP compiler version is {cann_version}; "
            f"expected {EXPECTED_CANN_COMPILER}"
        )

    distribution_versions = _distribution_versions(errors)
    vllm_version = distribution_versions.get("vllm")
    vllm_ascend_version = distribution_versions.get("vllm_ascend")
    torch_distribution_version = distribution_versions.get("torch")
    torch_npu_distribution_version = distribution_versions.get("torch_npu")
    (
        _torch_module,
        torch_npu_module,
        torch_version,
        torch_npu_version,
    ) = _dependency_versions(errors)
    runtime_fingerprint = _runtime_fingerprint(
        vllm_version=vllm_version,
        vllm_ascend_version=vllm_ascend_version,
        torch_version=torch_version,
        torch_npu_version=torch_npu_version,
        torch_distribution_version=torch_distribution_version,
        torch_npu_distribution_version=torch_npu_distribution_version,
        errors=errors,
    )
    soc_version = _device_soc_version(torch_npu_module)
    if soc_version is None:
        warnings.append(
            "Ascend SoC could not be detected before load; the bundled "
            "kernel supports only Ascend 910B-family devices"
        )
    elif not _soc_is_supported(soc_version):
        errors.append(
            f"unsupported Ascend SoC {soc_version!r}; "
            f"the bundled kernel targets {EXPECTED_SOC_FAMILY}"
        )

    return NativeStatus(
        enabled=True,
        loaded=False,
        compatible=not errors,
        state="compatible" if not errors else "incompatible",
        system=system,
        machine=machine,
        python_implementation=implementation,
        python_version=python_version,
        vllm_version=vllm_version,
        vllm_ascend_version=vllm_ascend_version,
        soc_version=soc_version,
        torch_version=torch_version,
        torch_npu_version=torch_npu_version,
        torch_distribution_version=torch_distribution_version,
        torch_npu_distribution_version=torch_npu_distribution_version,
        runtime_fingerprint=runtime_fingerprint,
        cann_compiler_version=cann_version,
        adapter_path=paths.adapter,
        opp_root=paths.opp_root,
        opp_vendor_path=paths.opp_vendor,
        cust_opapi_path=paths.cust_opapi,
        errors=tuple(errors),
        warnings=tuple(warnings),
    )


def _prepend_env_path(name: str, path: str) -> None:
    resolved = os.path.realpath(path)
    existing = [
        item
        for item in os.environ.get(name, "").split(os.pathsep)
        if item
    ]
    filtered = [
        item
        for item in existing
        if os.path.realpath(os.path.expanduser(item)) != resolved
    ]
    os.environ[name] = os.pathsep.join((resolved, *filtered))


def _configure_native_environment(status: NativeStatus) -> None:
    assert status.opp_vendor_path is not None
    assert status.cust_opapi_path is not None
    _prepend_env_path("ASCEND_CUSTOM_OPP_PATH", status.opp_vendor_path)
    _prepend_env_path(
        "LD_LIBRARY_PATH", os.path.dirname(status.cust_opapi_path)
    )


def _preload_cust_opapi(path: str) -> ctypes.CDLL:
    mode = getattr(ctypes, "RTLD_GLOBAL", os.RTLD_GLOBAL)
    return ctypes.CDLL(path, mode=mode)


def _load_torch_adapter(path: str) -> None:
    # Keep the adapter's existing one-path/idempotency checks in one place.
    kernel = importlib.import_module("vllm_ascend_trianglemix.kernel")
    kernel.load_triangle_mix_adapter(path)


def ensure_native_loaded(
    enabled: bool = True,
    *,
    strict: bool = False,
) -> NativeStatus:
    """Load the bundled OPP and Torch adapter once when explicitly enabled.

    ``strict=False`` converts all compatibility/load failures into a status
    with ``loaded=False`` so callers can fall back to official FIA.
    ``strict=True`` raises :class:`NativeBootstrapError` with the same status.
    """
    global _CUST_OPAPI_HANDLE, _LAST_STATUS
    with _LOAD_LOCK:
        if not enabled:
            disabled = inspect_native_compatibility(enabled=False)
            if not _LAST_STATUS.loaded:
                _LAST_STATUS = disabled
            return disabled
        if _LAST_STATUS.loaded:
            return _LAST_STATUS

        status = inspect_native_compatibility(enabled=True)
        if not status.compatible:
            _LAST_STATUS = status
            if strict:
                raise NativeBootstrapError(status)
            return status

        old_opp = os.environ.get("ASCEND_CUSTOM_OPP_PATH")
        old_ld = os.environ.get("LD_LIBRARY_PATH")
        try:
            _configure_native_environment(status)
            assert status.cust_opapi_path is not None
            assert status.adapter_path is not None
            cust_opapi_handle = _preload_cust_opapi(
                status.cust_opapi_path
            )
            _load_torch_adapter(status.adapter_path)
            _CUST_OPAPI_HANDLE = cust_opapi_handle
        except Exception as exc:
            if old_opp is None:
                os.environ.pop("ASCEND_CUSTOM_OPP_PATH", None)
            else:
                os.environ["ASCEND_CUSTOM_OPP_PATH"] = old_opp
            if old_ld is None:
                os.environ.pop("LD_LIBRARY_PATH", None)
            else:
                os.environ["LD_LIBRARY_PATH"] = old_ld
            failed = replace(
                status,
                loaded=False,
                compatible=False,
                state="load_failed",
                errors=status.errors
                + (
                    "native libraries could not be loaded "
                    f"({type(exc).__name__}: {exc}); detected vllm="
                    f"{status.vllm_version}, vllm-ascend="
                    f"{status.vllm_ascend_version}, torch="
                    f"{status.torch_version}, torch_npu="
                    f"{status.torch_npu_version}, SoC="
                    f"{status.soc_version or 'unknown'}",
                ),
            )
            _LAST_STATUS = failed
            if strict:
                raise NativeBootstrapError(failed) from exc
            return failed

        ready = replace(
            status,
            loaded=True,
            compatible=True,
            state="ready",
        )
        _LAST_STATUS = ready
        return ready


__all__ = [
    "NativeBootstrapError",
    "NativePaths",
    "NativeStatus",
    "SUPPORTED_RUNTIME_FINGERPRINTS",
    "ensure_native_loaded",
    "inspect_native_compatibility",
    "native_status",
]
