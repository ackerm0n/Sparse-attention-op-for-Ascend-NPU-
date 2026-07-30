#!/usr/bin/env python3
"""CPU-only tests for release adapter and wheel hygiene gates."""

from __future__ import annotations

import ast
import importlib.util
import io
import json
import pathlib
import sys
import tempfile
import unittest
import zipfile


ROOT = pathlib.Path(__file__).resolve().parents[1]
PROJECT_ROOT = ROOT.parent
CHECKER_PATH = ROOT / "tools/check_release_artifact.py"
SPEC = importlib.util.spec_from_file_location(
    "trianglemix_release_hygiene", CHECKER_PATH
)
assert SPEC is not None and SPEC.loader is not None
CHECKER = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = CHECKER
SPEC.loader.exec_module(CHECKER)


def minimal_adapter(*, machine: int = 183, debug: bool = False) -> bytes:
    """Build a minimal, structurally valid AArch64 ET_DYN fixture."""
    names_list = ["", ".text", ".shstrtab"]
    if debug:
        names_list.append(".debug_info")
    names = b"\0"
    offsets = {"": 0}
    for name in names_list[1:]:
        offsets[name] = len(names)
        names += name.encode("ascii") + b"\0"

    section_count = len(names_list)
    section_offset = CHECKER._ELF64_HEADER.size
    names_offset = (
        section_offset + section_count * CHECKER._ELF64_SECTION.size
    )
    ident = (
        CHECKER.ELF_MAGIC
        + bytes((CHECKER.ELFCLASS64, CHECKER.ELFDATA2LSB, 1, 0))
        + bytes(8)
    )
    header = CHECKER._ELF64_HEADER.pack(
        ident,
        CHECKER.ET_DYN,
        machine,
        1,
        0,
        0,
        section_offset,
        0,
        CHECKER._ELF64_HEADER.size,
        0,
        0,
        CHECKER._ELF64_SECTION.size,
        section_count,
        2,
    )
    sections = [bytes(CHECKER._ELF64_SECTION.size)]
    sections.append(
        CHECKER._ELF64_SECTION.pack(
            offsets[".text"], 1, 0, 0, names_offset + len(names), 0, 0, 0, 1, 0
        )
    )
    sections.append(
        CHECKER._ELF64_SECTION.pack(
            offsets[".shstrtab"],
            3,
            0,
            0,
            names_offset,
            len(names),
            0,
            0,
            1,
            0,
        )
    )
    if debug:
        sections.append(
            CHECKER._ELF64_SECTION.pack(
                offsets[".debug_info"],
                1,
                0,
                0,
                names_offset + len(names),
                0,
                0,
                0,
                1,
                0,
            )
        )
    return header + b"".join(sections) + names


def wheel_bytes(
    adapter: bytes,
    *,
    extra_entries: tuple[str, ...] = (),
) -> bytes:
    output = io.BytesIO()
    with zipfile.ZipFile(output, "w") as archive:
        archive.writestr(
            "vllm_ascend_trianglemix/_native/"
            "triangle_paged_attention_torch.cpython-310-"
            "aarch64-linux-gnu.so",
            adapter,
        )
        archive.writestr(
            "vllm_ascend_trianglemix/native.py",
            b"# release fixture\n",
        )
        for entry in extra_entries:
            archive.writestr(entry, b"# forbidden upstream override\n")
    return output.getvalue()


class ReleaseHygieneTests(unittest.TestCase):
    def test_minimal_release_adapter_passes(self) -> None:
        self.assertEqual(
            CHECKER.validate_adapter_bytes(
                minimal_adapter(), context="fixture.so"
            ),
            [],
        )

    def test_wrong_arch_and_debug_sections_fail(self) -> None:
        errors = CHECKER.validate_adapter_bytes(
            minimal_adapter(machine=62, debug=True),
            context="fixture.so",
        )
        details = " ".join(errors)
        self.assertIn("expected AArch64", details)
        self.assertIn(".debug_info", details)

    def test_every_forbidden_build_path_is_detected(self) -> None:
        for marker in CHECKER.FORBIDDEN_MARKERS:
            errors = CHECKER.validate_adapter_bytes(
                minimal_adapter() + marker + b"private/build/path",
                context="fixture.so",
            )
            self.assertTrue(errors, marker)
            self.assertIn("forbidden build-path marker", errors[0])

    def test_wheel_payload_is_scanned_not_just_filenames(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = pathlib.Path(directory) / "good.whl"
            path.write_bytes(wheel_bytes(minimal_adapter()))
            self.assertEqual(CHECKER.validate_wheel(path), [])

            path.write_bytes(
                wheel_bytes(minimal_adapter() + b"/mnt/private/source")
            )
            errors = CHECKER.validate_wheel(path)
            self.assertTrue(
                any("forbidden build-path marker" in item for item in errors)
            )

    def test_wheel_rejects_every_official_package_override(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = pathlib.Path(directory) / "candidate.whl"
            for entry in (
                "vllm/model_executor/layers/attention.py",
                "vllm_ascend/attention/attention_v1.py",
            ):
                with self.subTest(entry=entry):
                    path.write_bytes(
                        wheel_bytes(
                            minimal_adapter(),
                            extra_entries=(entry,),
                        )
                    )
                    errors = CHECKER.validate_wheel(path)
                    self.assertTrue(
                        any(
                            "official top-level package" in item
                            for item in errors
                        ),
                        errors,
                    )

    def test_candidate_contains_no_legacy_upstream_override_tree(self) -> None:
        self.assertFalse(
            (PROJECT_ROOT / "vllm_integration").exists(),
            "legacy direct site-packages overrides must stay outside "
            "the release candidate",
        )

    def test_build_recipe_has_reproducible_path_and_strip_gates(self) -> None:
        setup_source = (ROOT / "torch_adapter/setup.py").read_text()
        build_source = (ROOT / "torch_adapter/build.sh").read_text()
        for option in (
            "-ffile-prefix-map",
            "-fmacro-prefix-map",
            "-fdebug-prefix-map",
            "-g0",
        ):
            self.assertIn(option, setup_source)
        self.assertIn("--strip-unneeded", build_source)
        self.assertIn("check_release_artifact.py", build_source)
        for source in (setup_source, build_source):
            for marker in CHECKER.FORBIDDEN_MARKERS:
                self.assertNotIn(marker.decode("ascii"), source)

    def test_nested_prefix_maps_put_the_most_specific_path_last(self) -> None:
        setup_path = ROOT / "torch_adapter/setup.py"
        tree = ast.parse(setup_path.read_text(), filename=str(setup_path))
        helper = next(
            node
            for node in tree.body
            if isinstance(node, ast.FunctionDef)
            and node.name == "_ordered_prefix_mappings"
        )
        namespace = {"Path": pathlib.Path}
        exec(
            compile(
                ast.Module(body=[helper], type_ignores=[]),
                str(setup_path),
                "exec",
            ),
            namespace,
        )
        runtime = pathlib.Path("/build/python")
        site_root = runtime / "lib/python3.10/site-packages"
        ordered = namespace["_ordered_prefix_mappings"](
            {
                site_root: "/usr/src/python-deps/site-0",
                runtime: "/usr/src/python-runtime",
            }
        )
        self.assertEqual([source for source, _ in ordered], [runtime, site_root])

    def test_operator_preset_and_wheel_gate_are_host_independent(
        self,
    ) -> None:
        preset_path = ROOT / "CMakePresets.json"
        preset_source = preset_path.read_text()
        preset = json.loads(preset_source)
        cache = preset["configurePresets"][0]["cacheVariables"]
        self.assertEqual(
            cache["ASCEND_CANN_PACKAGE_PATH"]["value"],
            "$env{ASCEND_HOME_PATH}",
        )
        self.assertNotIn("CMAKE_CROSS_PLATFORM_COMPILER", cache)
        self.assertNotIn("/opt/", preset_source)

        package_setup = (
            PROJECT_ROOT / "package/setup.py"
        ).read_text()
        for release_gate in (
            "EM_AARCH64",
            "ET_DYN",
            "unstripped debug sections",
            "FORBIDDEN_TOP_LEVEL_PACKAGES",
            "ADAPTER_SUFFIX",
        ):
            self.assertIn(release_gate, package_setup)

    def test_source_distribution_license_and_authorship_are_explicit(
        self,
    ) -> None:
        canonical = (PROJECT_ROOT / "package/LICENSE").read_text()
        self.assertEqual(
            (PROJECT_ROOT / "LICENSE").read_text().strip(),
            canonical.strip(),
        )
        self.assertEqual(
            (
                PROJECT_ROOT / "licenses/Apache-2.0.txt"
            ).read_text().strip(),
            canonical.strip(),
        )
        for required in (
            PROJECT_ROOT / "NOTICE",
            PROJECT_ROOT / "THIRD_PARTY_NOTICES.md",
            PROJECT_ROOT / "licenses/CANN-OSL-2.0.txt",
        ):
            self.assertTrue(required.is_file(), required)

        original_files = (
            PROJECT_ROOT / "package/setup.py",
            PROJECT_ROOT
            / "package/tools/release_wheel_pipeline.py",
            PROJECT_ROOT
            / "package/src/vllm_ascend_trianglemix/kernel.py",
            PROJECT_ROOT
            / "package/src/vllm_ascend_trianglemix/native.py",
            ROOT / "torch_adapter/setup.py",
            ROOT / "torch_adapter/build.sh",
            ROOT / "torch_adapter/triangle_paged_attention_torch.cpp",
            ROOT / "tools/check_release_artifact.py",
        )
        for source in original_files:
            contents = source.read_text()
            self.assertIn("TriangleMix contributors", contents, source)
            self.assertNotIn(
                "Copyright (c) 2026 Huawei Technologies",
                contents,
                source,
            )


if __name__ == "__main__":
    unittest.main()
