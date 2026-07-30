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
"""Build the isolated Torch adapter against the active vLLM Ascend source."""

from __future__ import annotations

import os
import site
import sys
from pathlib import Path

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension
from torch_npu.utils.cpp_extension import NpuExtension


ADAPTER_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = ADAPTER_ROOT.parent
source_override = os.environ.get("VLLM_ASCEND_SRC")
if not source_override:
    raise RuntimeError(
        "Set VLLM_ASCEND_SRC to a vLLM-Ascend source checkout. The adapter "
        "build deliberately has no machine-specific source path fallback."
    )
VLLM_ASCEND_SRC = Path(source_override).expanduser().resolve()
C_SRC = VLLM_ASCEND_SRC / "csrc"

required = [
    C_SRC / "aclnn_torch_adapter/op_api_common.h",
    C_SRC / "aclnn_torch_adapter/NPUBridge.cpp",
    C_SRC / "aclnn_torch_adapter/NPUStorageImpl.cpp",
]
missing = [str(path) for path in required if not path.is_file()]
if missing:
    raise RuntimeError(f"missing vLLM Ascend adapter sources: {missing}")


def _ordered_prefix_mappings(
    mappings: dict[Path, str],
) -> list[tuple[Path, str]]:
    """Put nested paths after their parents for GCC's last-match rule."""
    return sorted(
        mappings.items(),
        key=lambda item: (len(str(item[0])), str(item[0])),
    )


def _prefix_map_flags() -> list[str]:
    """Hide build-host paths from debug data, macros, and diagnostics."""
    mappings = {
        PROJECT_ROOT: "/usr/src/trianglemix/operator",
        VLLM_ASCEND_SRC: "/usr/src/vllm-ascend",
        Path(sys.prefix).resolve(): "/usr/src/python-runtime",
    }
    site_roots = sorted(
        {
            Path(value).resolve()
            for value in site.getsitepackages()
            if Path(value).is_dir()
        },
        key=str,
    )
    for index, site_root in enumerate(site_roots):
        mappings[site_root] = f"/usr/src/python-deps/site-{index}"
    flags: list[str] = []
    # GCC applies the last matching prefix-map when source roots overlap.
    # Emit parents first so a nested site-packages map wins over sys.prefix.
    for source, target in _ordered_prefix_mappings(mappings):
        for option in (
            "-ffile-prefix-map",
            "-fmacro-prefix-map",
            "-fdebug-prefix-map",
        ):
            flags.append(f"{option}={source}={target}")
    return flags


compile_args = [
    "-O2",
    "-std=c++17",
    "-Wdate-time",
    *_prefix_map_flags(),
    # BuildExtension or distro flags may add -g before our extension flags.
    # Keep this last so no DWARF path table enters the release adapter.
    "-g0",
]


setup(
    name="triangle_paged_attention_torch",
    ext_modules=[
        NpuExtension(
            name="triangle_paged_attention_torch",
            sources=[
                str(ADAPTER_ROOT / "triangle_paged_attention_torch.cpp"),
                str(C_SRC / "aclnn_torch_adapter/NPUBridge.cpp"),
                str(C_SRC / "aclnn_torch_adapter/NPUStorageImpl.cpp"),
            ],
            include_dirs=[str(C_SRC)],
            extra_compile_args=compile_args,
            extra_link_args=["-Wl,--build-id=sha1"],
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)
