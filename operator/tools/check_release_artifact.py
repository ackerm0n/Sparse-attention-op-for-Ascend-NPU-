#!/usr/bin/env python3
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
"""Reject release artifacts that expose build hosts or an invalid adapter."""

from __future__ import annotations

import argparse
import struct
import sys
import zipfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Iterable


FORBIDDEN_MARKERS = (
    b"/mnt/",
    b"/Users/",
    b"siyuan.tong",
    b"/site-packages/",
)
FORBIDDEN_TOP_LEVEL_PACKAGES = frozenset(("vllm", "vllm_ascend"))
ADAPTER_BASENAME = "triangle_paged_attention_torch"
ELF_MAGIC = b"\x7fELF"
ELFCLASS64 = 2
ELFDATA2LSB = 1
ET_DYN = 3
EM_AARCH64 = 183

_ELF64_HEADER = struct.Struct("<16sHHIQQQIHHHHHH")
_ELF64_SECTION = struct.Struct("<IIQQQQIIQQ")


class ReleaseArtifactError(ValueError):
    """An artifact failed a deterministic release-hygiene check."""


@dataclass(frozen=True)
class ElfInfo:
    """Minimal ELF identity and section information used by the gate."""

    elf_type: int
    machine: int
    section_names: tuple[str, ...]


def forbidden_strings(
    data: bytes,
    *,
    context: str,
) -> list[str]:
    """Return stable diagnostics for forbidden path fragments in ``data``."""
    return [
        f"{context}: contains forbidden build-path marker "
        f"{marker.decode('ascii')!r}"
        for marker in FORBIDDEN_MARKERS
        if marker in data
    ]


def _bounded_slice(
    data: bytes,
    *,
    offset: int,
    size: int,
    context: str,
) -> bytes:
    if offset < 0 or size < 0 or offset > len(data) - size:
        raise ReleaseArtifactError(
            f"{context}: ELF table points outside the file"
        )
    return data[offset : offset + size]


def inspect_elf(data: bytes, *, context: str) -> ElfInfo:
    """Parse enough ELF64 metadata to verify the release adapter."""
    if len(data) < _ELF64_HEADER.size:
        raise ReleaseArtifactError(f"{context}: truncated ELF header")
    values = _ELF64_HEADER.unpack_from(data)
    ident = values[0]
    if ident[:4] != ELF_MAGIC:
        raise ReleaseArtifactError(f"{context}: not an ELF shared object")
    if ident[4] != ELFCLASS64 or ident[5] != ELFDATA2LSB:
        raise ReleaseArtifactError(
            f"{context}: expected a 64-bit little-endian ELF"
        )

    elf_type = values[1]
    machine = values[2]
    section_offset = values[6]
    section_entry_size = values[11]
    section_count = values[12]
    section_name_index = values[13]
    if section_entry_size < _ELF64_SECTION.size:
        raise ReleaseArtifactError(
            f"{context}: invalid ELF section entry size "
            f"{section_entry_size}"
        )
    if not section_count or section_name_index >= section_count:
        raise ReleaseArtifactError(f"{context}: invalid ELF section table")

    sections: list[tuple[int, int, int]] = []
    for index in range(section_count):
        entry_offset = section_offset + index * section_entry_size
        entry = _bounded_slice(
            data,
            offset=entry_offset,
            size=_ELF64_SECTION.size,
            context=context,
        )
        unpacked = _ELF64_SECTION.unpack(entry)
        sections.append((unpacked[0], unpacked[4], unpacked[5]))

    _, names_offset, names_size = sections[section_name_index]
    names = _bounded_slice(
        data,
        offset=names_offset,
        size=names_size,
        context=context,
    )
    section_names: list[str] = []
    for name_offset, _, _ in sections:
        if name_offset >= len(names):
            raise ReleaseArtifactError(
                f"{context}: invalid ELF section-name offset"
            )
        end = names.find(b"\0", name_offset)
        if end < 0:
            raise ReleaseArtifactError(
                f"{context}: unterminated ELF section name"
            )
        section_names.append(
            names[name_offset:end].decode("ascii", errors="replace")
        )
    return ElfInfo(
        elf_type=elf_type,
        machine=machine,
        section_names=tuple(section_names),
    )


def validate_adapter_bytes(data: bytes, *, context: str) -> list[str]:
    """Return every release-gate violation for one adapter image."""
    errors = forbidden_strings(data, context=context)
    try:
        elf = inspect_elf(data, context=context)
    except ReleaseArtifactError as exc:
        return [*errors, str(exc)]
    if elf.elf_type != ET_DYN:
        errors.append(
            f"{context}: expected ET_DYN ({ET_DYN}), got {elf.elf_type}"
        )
    if elf.machine != EM_AARCH64:
        errors.append(
            f"{context}: expected AArch64 e_machine={EM_AARCH64}, "
            f"got {elf.machine}"
        )
    debug_sections = sorted(
        name
        for name in elf.section_names
        if name.startswith((".debug", ".zdebug"))
        or name == ".gnu_debuglink"
    )
    if debug_sections:
        errors.append(
            f"{context}: unstripped debug sections: "
            + ", ".join(debug_sections)
        )
    return errors


def validate_adapter(path: Path) -> list[str]:
    return validate_adapter_bytes(path.read_bytes(), context=str(path))


def validate_wheel(path: Path) -> list[str]:
    """Reject upstream overrides and validate every release payload."""
    errors: list[str] = []
    adapters = 0
    try:
        with zipfile.ZipFile(path) as archive:
            for info in archive.infolist():
                archive_path = PurePosixPath(info.filename)
                if (
                    archive_path.parts
                    and archive_path.parts[0]
                    in FORBIDDEN_TOP_LEVEL_PACKAGES
                ):
                    errors.append(
                        f"{path}:{info.filename}: would overwrite an "
                        "official top-level package"
                    )
                errors.extend(
                    forbidden_strings(
                        info.filename.encode("utf-8"),
                        context=f"{path}:{info.filename}",
                    )
                )
                if info.is_dir():
                    continue
                data = archive.read(info)
                context = f"{path}:{info.filename}"
                errors.extend(forbidden_strings(data, context=context))
                filename = Path(info.filename).name
                if (
                    filename.startswith(ADAPTER_BASENAME)
                    and filename.endswith(".so")
                ):
                    adapters += 1
                    errors.extend(
                        validate_adapter_bytes(data, context=context)
                    )
    except (OSError, zipfile.BadZipFile) as exc:
        return [f"{path}: unreadable wheel ({type(exc).__name__}: {exc})"]
    if adapters != 1:
        errors.append(
            f"{path}: expected exactly one {ADAPTER_BASENAME}*.so, "
            f"found {adapters}"
        )
    return errors


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--adapter",
        action="append",
        type=Path,
        default=[],
        help="stripped adapter ELF to validate (repeatable)",
    )
    parser.add_argument(
        "--wheel",
        action="append",
        type=Path,
        default=[],
        help="wheel to scan and validate (repeatable)",
    )
    return parser


def _all_errors(
    adapters: Iterable[Path],
    wheels: Iterable[Path],
) -> list[str]:
    errors: list[str] = []
    for path in adapters:
        errors.extend(validate_adapter(path))
    for path in wheels:
        errors.extend(validate_wheel(path))
    return errors


def main() -> int:
    args = _parser().parse_args()
    if not args.adapter and not args.wheel:
        print("at least one --adapter or --wheel is required", file=sys.stderr)
        return 2
    errors = _all_errors(args.adapter, args.wheel)
    if errors:
        for error in errors:
            print(error, file=sys.stderr)
        return 1
    checked = len(args.adapter) + len(args.wheel)
    print(f"release artifact validation passed: {checked} artifact(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
