#!/usr/bin/env python3
"""Static contracts for a self-contained dynamic-source OPP package."""

from __future__ import annotations

import pathlib
import re
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[1]
OP_KERNEL = (ROOT / "op_kernel").resolve()
VENDORED_BLOCK_SPARSE = (
    OP_KERNEL / "vendor/cann_9_0_1/block_sparse_attention"
)
QUOTED_INCLUDE = re.compile(r'^\s*#\s*include\s+"([^"]+)"', re.MULTILINE)


class PackageSelfContainedTest(unittest.TestCase):
    def test_production_include_uses_packaged_vendor_tree(self) -> None:
        paged_mmad = OP_KERNEL / "triangle_paged_block_mmad.h"
        source = paged_mmad.read_text()
        include = (
            "vendor/cann_9_0_1/block_sparse_attention/kernel_common.hpp"
        )
        self.assertIn(f'#include "{include}"', source)
        self.assertNotIn('#include "../vendor/', source)
        self.assertTrue((OP_KERNEL / include).is_file())

    def test_vendored_header_closure_is_present_and_licensed(self) -> None:
        headers = sorted(
            path
            for path in VENDORED_BLOCK_SPARSE.rglob("*")
            if path.suffix in {".h", ".hpp"}
        )
        self.assertGreaterEqual(len(headers), 51)
        for header in headers:
            self.assertIn(
                "CANN Open Software License Agreement Version 2.0",
                header.read_text(),
                str(header.relative_to(ROOT)),
            )

    def test_relative_includes_cannot_escape_dynamic_source_root(self) -> None:
        sources = sorted(
            path
            for path in OP_KERNEL.rglob("*")
            if path.suffix in {".h", ".hpp", ".cpp"}
        )
        for source in sources:
            for include in QUOTED_INCLUDE.findall(source.read_text()):
                if not include.startswith("."):
                    continue
                resolved = (source.parent / include).resolve()
                try:
                    resolved.relative_to(OP_KERNEL)
                except ValueError as error:
                    self.fail(
                        f"{source.relative_to(ROOT)} include {include!r} "
                        f"escapes op_kernel: {error}"
                    )
                self.assertTrue(
                    resolved.is_file(),
                    f"{source.relative_to(ROOT)} include {include!r} "
                    "is missing from the package source tree",
                )


if __name__ == "__main__":
    unittest.main()
