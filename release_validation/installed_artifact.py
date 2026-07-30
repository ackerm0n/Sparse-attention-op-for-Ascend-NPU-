"""Prove that imported TriangleMix files match one exact wheel artifact."""

from __future__ import annotations

import hashlib
import re
import zipfile
from email.parser import BytesParser
from importlib import metadata
from pathlib import Path, PurePosixPath
from typing import Any

from .common import is_relative_to, sha256_file, sha256_json


PACKAGE_PREFIX = "vllm_ascend_trianglemix/"
DISTRIBUTION_NAME = "vllm-ascend-trianglemix"


def _canonical_name(value: str) -> str:
    return re.sub(r"[-_.]+", "-", value).lower()


def _payload_sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def audit_installed_wheel(
    wheel_path: Path,
    *,
    distribution: metadata.Distribution | None = None,
) -> dict[str, Any]:
    """Compare every installed package payload with the supplied wheel.

    ``RECORD`` and other dist-info files are intentionally excluded because
    installers rewrite them. Every file under the import package, including
    the adapter and private OPP tree, must match byte-for-byte.
    """

    wheel_path = wheel_path.resolve()
    errors: list[dict[str, object]] = []
    if not wheel_path.is_file():
        return {
            "passed": False,
            "wheel": str(wheel_path),
            "errors": [{"kind": "missing_wheel"}],
        }
    dist = distribution or metadata.distribution(DISTRIBUTION_NAME)
    distribution_root = Path(dist.locate_file("")).resolve()
    files = dist.files
    if files is None:
        return {
            "passed": False,
            "wheel": str(wheel_path),
            "distribution_root": str(distribution_root),
            "errors": [{"kind": "missing_distribution_record"}],
        }

    recorded_entries = {
        str(file).replace("\\", "/")
        for file in files
        if str(file).replace("\\", "/").startswith(PACKAGE_PREFIX)
        and not str(file).endswith((".pyc", ".pyo"))
        and "/__pycache__/" not in str(file).replace("\\", "/")
    }
    package_root = Path(
        dist.locate_file(PACKAGE_PREFIX.rstrip("/"))
    ).resolve()
    installed_entries: set[str] = set()
    if not is_relative_to(package_root, distribution_root):
        errors.append(
            {
                "kind": "package_outside_distribution_root",
                "package_root": str(package_root),
            }
        )
    elif not package_root.is_dir():
        errors.append(
            {
                "kind": "missing_installed_package",
                "package_root": str(package_root),
            }
        )
    else:
        for installed_path in package_root.rglob("*"):
            if not (installed_path.is_file() or installed_path.is_symlink()):
                continue
            name = installed_path.relative_to(distribution_root).as_posix()
            if (
                name.endswith((".pyc", ".pyo"))
                or "/__pycache__/" in name
            ):
                continue
            installed_entries.add(name)
    unrecorded = sorted(installed_entries - recorded_entries)
    absent_from_disk = sorted(recorded_entries - installed_entries)
    if unrecorded:
        errors.append(
            {
                "kind": "installed_unrecorded_payload",
                "count": len(unrecorded),
                "entries": unrecorded[:32],
            }
        )
    if absent_from_disk:
        errors.append(
            {
                "kind": "recorded_payload_absent_from_disk",
                "count": len(absent_from_disk),
                "entries": absent_from_disk[:32],
            }
        )
    wheel_hashes: dict[str, str] = {}
    metadata_values: list[dict[str, str | None]] = []
    try:
        with zipfile.ZipFile(wheel_path) as archive:
            for info in archive.infolist():
                name = info.filename
                pure = PurePosixPath(name)
                if pure.is_absolute() or ".." in pure.parts:
                    errors.append(
                        {"kind": "unsafe_wheel_entry", "entry": name}
                    )
                    continue
                if name.endswith(".dist-info/METADATA") and not info.is_dir():
                    parsed = BytesParser().parsebytes(archive.read(info))
                    metadata_values.append(
                        {
                            "entry": name,
                            "name": parsed.get("Name"),
                            "version": parsed.get("Version"),
                        }
                    )
                if (
                    name.startswith(PACKAGE_PREFIX)
                    and not info.is_dir()
                    and not name.endswith((".pyc", ".pyo"))
                    and "/__pycache__/" not in name
                ):
                    if name in wheel_hashes:
                        errors.append(
                            {
                                "kind": "duplicate_wheel_payload",
                                "entry": name,
                            }
                        )
                    wheel_hashes[name] = _payload_sha256(archive.read(info))
    except (OSError, zipfile.BadZipFile) as error:
        errors.append(
            {
                "kind": "unreadable_wheel",
                "type": type(error).__name__,
                "message": str(error),
            }
        )

    if len(metadata_values) != 1:
        errors.append(
            {
                "kind": "wheel_metadata_count",
                "observed": len(metadata_values),
            }
        )
    else:
        wheel_metadata = metadata_values[0]
        installed_name = str(dist.metadata.get("Name") or "")
        if (
            _canonical_name(str(wheel_metadata["name"] or ""))
            != _canonical_name(installed_name)
            or _canonical_name(installed_name)
            != _canonical_name(DISTRIBUTION_NAME)
        ):
            errors.append(
                {
                    "kind": "distribution_name_mismatch",
                    "wheel": wheel_metadata["name"],
                    "installed": installed_name,
                }
            )
        if str(wheel_metadata["version"]) != str(dist.version):
            errors.append(
                {
                    "kind": "distribution_version_mismatch",
                    "wheel": wheel_metadata["version"],
                    "installed": dist.version,
                }
            )

    wheel_entries = set(wheel_hashes)
    missing = sorted(wheel_entries - installed_entries)
    unexpected = sorted(installed_entries - wheel_entries)
    if missing:
        errors.append(
            {
                "kind": "installed_payload_missing",
                "count": len(missing),
                "entries": missing[:32],
            }
        )
    if unexpected:
        errors.append(
            {
                "kind": "installed_payload_unexpected",
                "count": len(unexpected),
                "entries": unexpected[:32],
            }
        )

    mismatches: list[dict[str, str]] = []
    installed_hashes: dict[str, str] = {}
    for name in sorted(wheel_entries & installed_entries):
        installed_path = Path(dist.locate_file(name)).resolve()
        if not is_relative_to(installed_path, distribution_root):
            mismatches.append(
                {
                    "entry": name,
                    "kind": "outside_distribution_root",
                    "installed_path": str(installed_path),
                }
            )
            continue
        if not installed_path.is_file():
            mismatches.append(
                {
                    "entry": name,
                    "kind": "not_a_regular_file",
                    "installed_path": str(installed_path),
                }
            )
            continue
        installed_hash = sha256_file(installed_path)
        installed_hashes[name] = installed_hash
        if installed_hash != wheel_hashes[name]:
            mismatches.append(
                {
                    "entry": name,
                    "kind": "sha256_mismatch",
                    "wheel_sha256": wheel_hashes[name],
                    "installed_sha256": installed_hash,
                }
            )
    if mismatches:
        errors.append(
            {
                "kind": "installed_payload_mismatch",
                "count": len(mismatches),
                "entries": mismatches[:32],
            }
        )

    return {
        "passed": not errors,
        "wheel": str(wheel_path),
        "wheel_sha256": sha256_file(wheel_path),
        "distribution_name": dist.metadata.get("Name"),
        "distribution_version": dist.version,
        "distribution_root": str(distribution_root),
        "package_payload_count": len(wheel_hashes),
        "installed_payload_count": len(installed_entries),
        "recorded_payload_count": len(recorded_entries),
        "wheel_payload_manifest_sha256": sha256_json(wheel_hashes),
        "installed_payload_manifest_sha256": sha256_json(installed_hashes),
        "errors": errors,
    }
