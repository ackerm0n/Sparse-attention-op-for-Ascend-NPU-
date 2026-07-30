from __future__ import annotations

import importlib.util
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
SETUP_PATH = PACKAGE_ROOT / "setup.py"


def load_setup_module():
    name = f"_trianglemix_setup_test_{id(object())}"
    spec = importlib.util.spec_from_file_location(name, SETUP_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    with patch("setuptools.setup"):
        spec.loader.exec_module(module)
    return module


def minimal_adapter(
    setup_module,
    *,
    machine: int | None = None,
    debug: bool = False,
) -> bytes:
    names_list = ["", ".text", ".shstrtab"]
    if debug:
        names_list.append(".debug_info")
    names = b"\0"
    offsets = {"": 0}
    for name in names_list[1:]:
        offsets[name] = len(names)
        names += name.encode("ascii") + b"\0"

    section_count = len(names_list)
    section_offset = setup_module._ELF64_HEADER.size
    names_offset = (
        section_offset
        + section_count * setup_module._ELF64_SECTION.size
    )
    ident = (
        setup_module.ELF_MAGIC
        + bytes(
            (
                setup_module.ELFCLASS64,
                setup_module.ELFDATA2LSB,
                1,
                0,
            )
        )
        + bytes(8)
    )
    header = setup_module._ELF64_HEADER.pack(
        ident,
        setup_module.ET_DYN,
        setup_module.EM_AARCH64 if machine is None else machine,
        1,
        0,
        0,
        section_offset,
        0,
        setup_module._ELF64_HEADER.size,
        0,
        0,
        setup_module._ELF64_SECTION.size,
        section_count,
        2,
    )
    sections = [bytes(setup_module._ELF64_SECTION.size)]
    sections.append(
        setup_module._ELF64_SECTION.pack(
            offsets[".text"],
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
    sections.append(
        setup_module._ELF64_SECTION.pack(
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
            setup_module._ELF64_SECTION.pack(
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


def write_wheel(
    setup_module,
    path: Path,
    adapter: bytes,
    *,
    extra_entry: str | None = None,
) -> None:
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr(
            "vllm_ascend_trianglemix/_native/"
            f"{setup_module.ADAPTER_BASENAME}"
            f"{setup_module.ADAPTER_SUFFIX}",
            adapter,
        )
        if extra_entry is not None:
            archive.writestr(extra_entry, b"# must be rejected\n")


class BuildHygieneTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.setup_module = load_setup_module()

    @classmethod
    def tearDownClass(cls) -> None:
        for name in tuple(sys.modules):
            if name.startswith("_trianglemix_setup_test_"):
                sys.modules.pop(name, None)

    def test_release_adapter_requires_aarch64_and_no_debug_sections(
        self,
    ) -> None:
        module = self.setup_module
        self.assertEqual(
            module._adapter_errors(
                minimal_adapter(module),
                context="adapter.so",
            ),
            [],
        )
        errors = module._adapter_errors(
            minimal_adapter(module, machine=62, debug=True),
            context="adapter.so",
        )
        details = " ".join(errors)
        self.assertIn("expected AArch64", details)
        self.assertIn(".debug_info", details)

    def test_wheel_rejects_official_package_overrides(self) -> None:
        module = self.setup_module
        with tempfile.TemporaryDirectory() as directory:
            wheel = Path(directory) / "candidate.whl"
            write_wheel(
                module,
                wheel,
                minimal_adapter(module),
                extra_entry="vllm_ascend/attention/attention_v1.py",
            )
            errors = module._wheel_path_errors(wheel)
        self.assertTrue(
            any("official top-level package" in item for item in errors)
        )

    def test_wheel_accepts_one_stripped_plugin_adapter(self) -> None:
        module = self.setup_module
        with tempfile.TemporaryDirectory() as directory:
            wheel = Path(directory) / "candidate.whl"
            write_wheel(module, wheel, minimal_adapter(module))
            self.assertEqual(module._wheel_path_errors(wheel), [])

    def test_wheel_tag_is_exact_cp310_linux_aarch64(self) -> None:
        module = self.setup_module
        command = module.PrebuiltAarch64Wheel(
            module.BinaryDistribution()
        )
        command.ensure_finalized()
        self.assertEqual(
            command.get_tag(),
            ("cp310", "cp310", "linux_aarch64"),
        )


if __name__ == "__main__":
    unittest.main()
