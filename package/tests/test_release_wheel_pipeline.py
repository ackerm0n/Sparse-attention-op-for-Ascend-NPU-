from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import types
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
PIPELINE_PATH = (
    PACKAGE_ROOT / "tools" / "release_wheel_pipeline.py"
)


def load_pipeline_module():
    name = f"_trianglemix_release_pipeline_test_{id(object())}"
    spec = importlib.util.spec_from_file_location(name, PIPELINE_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


class ReleaseWheelPipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.pipeline = load_pipeline_module()

    @classmethod
    def tearDownClass(cls) -> None:
        for name in tuple(sys.modules):
            if name.startswith(
                "_trianglemix_release_pipeline_test_"
            ):
                sys.modules.pop(name, None)

    def test_manifest_detects_content_and_mode_changes(self) -> None:
        pipeline = self.pipeline
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "module.py"
            source.write_text("first\n", encoding="utf-8")
            source.chmod(0o640)
            first = pipeline._tree_manifest(root)
            source.write_text("second\n", encoding="utf-8")
            source.chmod(0o600)
            second = pipeline._tree_manifest(root)
        self.assertNotEqual(first, second)
        self.assertNotEqual(
            pipeline._manifest_digest(first),
            pipeline._manifest_digest(second),
        )

    def test_wheel_root_gate_rejects_upstream_payload(self) -> None:
        pipeline = self.pipeline
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "candidate.whl"
            with zipfile.ZipFile(path, "w") as archive:
                archive.writestr(
                    "vllm_ascend_trianglemix/plugin.py",
                    b"# plugin\n",
                )
                archive.writestr(
                    "vllm_ascend/attention/attention_v1.py",
                    b"# forbidden override\n",
                )
            with self.assertRaisesRegex(
                pipeline.PipelineError,
                "unexpected top-level payloads",
            ):
                pipeline._validate_wheel_roots(path)

    def test_output_gate_never_overwrites_existing_wheel(self) -> None:
        pipeline = self.pipeline
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            existing = (
                output
                / "vllm_ascend_trianglemix-0.1.0-"
                "cp310-cp310-linux_aarch64.whl"
            )
            existing.write_bytes(b"do not overwrite")
            with self.assertRaisesRegex(
                pipeline.PipelineError,
                "refusing to overwrite",
            ):
                pipeline._require_fresh_output(output)
            self.assertEqual(existing.read_bytes(), b"do not overwrite")

    def test_preflight_rejects_preinstalled_plugin(self) -> None:
        pipeline = self.pipeline
        installed = types.SimpleNamespace(version="0.0-test")
        with (
            patch.object(
                pipeline.metadata,
                "distribution",
                return_value=installed,
            ),
            self.assertRaisesRegex(
                pipeline.PipelineError,
                "base environment already contains",
            ),
        ):
            pipeline._require_plugin_absent()

    def test_cann_install_path_script_is_rewritten_portably(self) -> None:
        pipeline = self.pipeline
        with tempfile.TemporaryDirectory() as directory:
            opp_root = Path(directory) / "opp"
            script = (
                opp_root
                / "vendors"
                / "trianglemix"
                / "bin"
                / "set_env.bash"
            )
            script.parent.mkdir(parents=True)
            script.write_text(
                "#!/bin/bash\n"
                "export ASCEND_CUSTOM_OPP_PATH=/mnt/private/build\n",
                encoding="utf-8",
            )
            rewritten = pipeline._write_portable_opp_set_env(opp_root)

            self.assertEqual(rewritten, script)
            content = script.read_text(encoding="utf-8")
            self.assertEqual(content, pipeline.PORTABLE_OPP_SET_ENV)
            self.assertIn("${BASH_SOURCE[0]}", content)
            self.assertNotIn("/mnt/", content)
            self.assertNotIn("siyuan.tong", content)
            self.assertEqual(script.stat().st_mode & 0o777, 0o755)

    def test_venv_site_packages_uses_one_existing_purelib(self) -> None:
        pipeline = self.pipeline
        with tempfile.TemporaryDirectory() as directory:
            venv = Path(directory) / "venv"
            python = venv / "bin" / "python"
            purelib = venv / "lib" / "python3.10" / "site-packages"
            python.parent.mkdir(parents=True)
            purelib.mkdir(parents=True)
            completed = types.SimpleNamespace(
                stdout=json.dumps(str(purelib)) + "\n"
            )
            with patch.object(pipeline, "_run", return_value=completed):
                self.assertEqual(
                    pipeline._venv_site_packages(python, {}),
                    purelib.resolve(),
                )

            external = Path(directory) / "outside"
            external.mkdir()
            completed.stdout = json.dumps(str(external)) + "\n"
            with (
                patch.object(pipeline, "_run", return_value=completed),
                self.assertRaisesRegex(
                    pipeline.PipelineError,
                    "cannot identify",
                ),
            ):
                pipeline._venv_site_packages(python, {})

    def test_clean_venv_inherits_only_explicit_existing_roots(self) -> None:
        pipeline = self.pipeline
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            site_packages = root / "venv" / "site-packages"
            first = root / "base-runtime"
            second = root / "official-source"
            site_packages.mkdir(parents=True)
            first.mkdir()
            second.mkdir()

            pth = pipeline._write_base_environment_pth(
                site_packages,
                (second, first, second),
            )
            self.assertEqual(
                pth.read_text(encoding="utf-8").splitlines(),
                sorted((str(first.resolve()), str(second.resolve()))),
            )

            missing = root / "missing"
            with self.assertRaisesRegex(
                pipeline.PipelineError,
                "invalid clean-install inheritance",
            ):
                pipeline._write_base_environment_pth(
                    site_packages,
                    (missing,),
                )

    def test_real_environment_smoke_uses_official_import_order(self) -> None:
        source = (
            PACKAGE_ROOT / "tests" / "real_env_smoke.py"
        ).read_text(encoding="utf-8")
        ops_import = (
            "importlib.import_module(ASCEND_OPS_MODULE)"
        )
        attention_import = (
            "importlib.import_module(ATTENTION_MODULE)"
        )
        self.assertIn(ops_import, source)
        self.assertIn(attention_import, source)
        self.assertLess(
            source.index(ops_import),
            source.index(attention_import),
        )


if __name__ == "__main__":
    unittest.main()
