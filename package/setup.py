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
"""Build the wheel around an existing CPython 3.10/AArch64 adapter."""

from __future__ import annotations

import struct
import zipfile
from distutils.errors import DistutilsSetupError
from pathlib import Path, PurePosixPath

from setuptools import find_packages, setup
from setuptools.dist import Distribution
from wheel.bdist_wheel import bdist_wheel


ROOT = Path(__file__).resolve().parent
PROJECT_README = ROOT.parent / "README.md"
PACKAGE_NAME = "vllm_ascend_trianglemix"
PACKAGE_DIR = ROOT / "src" / PACKAGE_NAME
NATIVE_DIR = PACKAGE_DIR / "_native"
FORBIDDEN_WHEEL_MARKERS = (
    b"/mnt/",
    b"/Users/",
    b"siyuan.tong",
    b"/site-packages/",
)
FORBIDDEN_TOP_LEVEL_PACKAGES = frozenset(("vllm", "vllm_ascend"))
ADAPTER_BASENAME = "triangle_paged_attention_torch"
ADAPTER_SUFFIX = ".cpython-310-aarch64-linux-gnu.so"
ELF_MAGIC = b"\x7fELF"
ELFCLASS64 = 2
ELFDATA2LSB = 1
ET_DYN = 3
EM_AARCH64 = 183

_ELF64_HEADER = struct.Struct("<16sHHIQQQIHHHHHH")
_ELF64_SECTION = struct.Struct("<IIQQQQIIQQ")


class BinaryDistribution(Distribution):
    """Mark the distribution as platform-specific without compiling code."""

    def has_ext_modules(self) -> bool:
        return True

    def is_pure(self) -> bool:
        return False


class PrebuiltAarch64Wheel(bdist_wheel):
    """Tag the wheel for the exact ABI of the bundled adapter."""

    def finalize_options(self) -> None:
        super().finalize_options()
        self.root_is_pure = False

    def get_tag(self) -> tuple[str, str, str]:
        return ("cp310", "cp310", "linux_aarch64")

    def run(self) -> None:
        output = Path(self.dist_dir).resolve()
        before = {
            path.resolve(): path.stat().st_mtime_ns
            for path in output.glob("*.whl")
        } if output.is_dir() else {}
        super().run()
        candidates = [
            path
            for path in output.glob("*.whl")
            if before.get(path.resolve()) != path.stat().st_mtime_ns
        ]
        if not candidates:
            raise DistutilsSetupError(
                "bdist_wheel did not produce a verifiable wheel"
            )
        for wheel_path in candidates:
            errors = _wheel_path_errors(wheel_path)
            if errors:
                # Never leave a path-leaking file looking like a release
                # artifact after a failed build command.
                wheel_path.unlink(missing_ok=True)
                raise DistutilsSetupError(
                    "release wheel hygiene check failed:\n"
                    + "\n".join(errors)
                )


def _native_package_data() -> list[str]:
    return [
        path.relative_to(PACKAGE_DIR).as_posix()
        for path in sorted(NATIVE_DIR.rglob("*"))
        if path.is_file()
    ]


def _bounded_slice(
    data: bytes,
    *,
    offset: int,
    size: int,
    context: str,
) -> bytes:
    if offset < 0 or size < 0 or offset > len(data) - size:
        raise ValueError(f"{context}: ELF table points outside the file")
    return data[offset : offset + size]


def _adapter_errors(data: bytes, *, context: str) -> list[str]:
    """Validate the release adapter without relying on host ELF tools."""
    if len(data) < _ELF64_HEADER.size:
        return [f"{context}: truncated ELF header"]
    values = _ELF64_HEADER.unpack_from(data)
    ident = values[0]
    if ident[:4] != ELF_MAGIC:
        return [f"{context}: not an ELF shared object"]
    if ident[4] != ELFCLASS64 or ident[5] != ELFDATA2LSB:
        return [f"{context}: expected a 64-bit little-endian ELF"]

    errors: list[str] = []
    if values[1] != ET_DYN:
        errors.append(
            f"{context}: expected ET_DYN ({ET_DYN}), got {values[1]}"
        )
    if values[2] != EM_AARCH64:
        errors.append(
            f"{context}: expected AArch64 e_machine={EM_AARCH64}, "
            f"got {values[2]}"
        )

    section_offset = values[6]
    section_entry_size = values[11]
    section_count = values[12]
    section_name_index = values[13]
    if section_entry_size < _ELF64_SECTION.size:
        return [
            *errors,
            f"{context}: invalid ELF section entry size "
            f"{section_entry_size}",
        ]
    if not section_count or section_name_index >= section_count:
        return [*errors, f"{context}: invalid ELF section table"]

    try:
        sections: list[tuple[int, int, int]] = []
        for index in range(section_count):
            entry = _bounded_slice(
                data,
                offset=section_offset + index * section_entry_size,
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
                raise ValueError(
                    f"{context}: invalid ELF section-name offset"
                )
            end = names.find(b"\0", name_offset)
            if end < 0:
                raise ValueError(
                    f"{context}: unterminated ELF section name"
                )
            section_names.append(
                names[name_offset:end].decode(
                    "ascii",
                    errors="replace",
                )
            )
    except ValueError as exc:
        return [*errors, str(exc)]

    debug_sections = sorted(
        name
        for name in section_names
        if name.startswith((".debug", ".zdebug"))
        or name == ".gnu_debuglink"
    )
    if debug_sections:
        errors.append(
            f"{context}: unstripped debug sections: "
            + ", ".join(debug_sections)
        )
    return errors


def _wheel_path_errors(path: Path) -> list[str]:
    """Reject path leaks, upstream overrides, and an invalid adapter."""
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
                        f"{info.filename} would overwrite an official "
                        "top-level package"
                    )
                values = (
                    ("entry name", info.filename.encode("utf-8")),
                    ("payload", b"" if info.is_dir() else archive.read(info)),
                )
                for value_name, value in values:
                    for marker in FORBIDDEN_WHEEL_MARKERS:
                        if marker in value:
                            errors.append(
                                f"{info.filename} {value_name} contains "
                                f"{marker.decode('ascii')!r}"
                            )
                filename = Path(info.filename).name
                if (
                    filename.startswith(ADAPTER_BASENAME)
                    and filename.endswith(".so")
                ):
                    adapters += 1
                    if not filename.endswith(ADAPTER_SUFFIX):
                        errors.append(
                            f"{info.filename} does not use the required "
                            f"CPython 3.10/AArch64 suffix {ADAPTER_SUFFIX!r}"
                        )
                    errors.extend(
                        _adapter_errors(
                            values[1][1],
                            context=info.filename,
                        )
                    )
    except (OSError, zipfile.BadZipFile) as exc:
        errors.append(
            f"{path} is not a readable wheel "
            f"({type(exc).__name__}: {exc})"
        )
    if adapters != 1:
        errors.append(
            "wheel must contain exactly one "
            "triangle_paged_attention_torch*.so "
            f"(found {adapters})"
        )
    return errors


setup(
    name="vllm-ascend-trianglemix",
    version="0.1.0",
    description=(
        "Single-launch paged-KV TriangleMix sparse prefill for vLLM-Ascend"
    ),
    long_description=PROJECT_README.read_text(encoding="utf-8"),
    long_description_content_type="text/markdown",
    author="TriangleMix contributors",
    # Python integration is Apache-2.0. The bundled Huawei CANN material is
    # governed by the separately shipped CANN OSL 2.0 terms.
    license="Apache-2.0 AND LicenseRef-CANN-OSL-2.0",
    license_files=(
        "LICENSE",
        "NOTICE",
        "THIRD_PARTY_NOTICES.md",
        "licenses/*.txt",
    ),
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    package_data={PACKAGE_NAME: _native_package_data()},
    include_package_data=True,
    install_requires=("packaging>=23",),
    python_requires=">=3.10,<3.11",
    entry_points={
        "vllm.general_plugins": [
            "trianglemix = vllm_ascend_trianglemix.plugin:register",
        ],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Environment :: GPU",
        "Intended Audience :: Developers",
        "Operating System :: POSIX :: Linux",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    zip_safe=False,
    distclass=BinaryDistribution,
    cmdclass={"bdist_wheel": PrebuiltAarch64Wheel},
)
