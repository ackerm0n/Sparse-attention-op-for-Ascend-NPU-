from __future__ import annotations

import ctypes
import importlib.util
import os
import sys
import unittest
from pathlib import Path
from unittest.mock import Mock, patch


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
NATIVE_SOURCE = (
    PACKAGE_ROOT / "src" / "vllm_ascend_trianglemix" / "native.py"
)


def load_native_module():
    """Load native.py without importing the package or its torch modules."""
    name = f"_trianglemix_native_test_{id(object())}"
    spec = importlib.util.spec_from_file_location(name, NATIVE_SOURCE)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


class NativeBootstrapTests(unittest.TestCase):
    def setUp(self) -> None:
        self.native = load_native_module()

    def tearDown(self) -> None:
        for name in tuple(sys.modules):
            if name.startswith("_trianglemix_native_test_"):
                sys.modules.pop(name, None)

    def compatible_status(self):
        paths = self.native._resolve_bundled_paths()
        return self.native.NativeStatus(
            enabled=True,
            loaded=False,
            compatible=True,
            state="compatible",
            system="Linux",
            machine="aarch64",
            python_implementation="cpython",
            python_version="3.10.17",
            vllm_version="0.23.0+empty",
            vllm_ascend_version="0.23.0rc1",
            soc_version="Ascend910B2",
            torch_version="2.10.0+cpu",
            torch_npu_version="2.10.0.post2",
            torch_distribution_version="2.10.0",
            torch_npu_distribution_version="2.10.0.post2",
            runtime_fingerprint=(
                "0.23.0",
                "0.23.0rc1",
                "2.10.0",
                "2.10.0.post2",
            ),
            cann_compiler_version="9.0.1",
            adapter_path=paths.adapter,
            opp_root=paths.opp_root,
            opp_vendor_path=paths.opp_vendor,
            cust_opapi_path=paths.cust_opapi,
        )

    def test_disabled_has_no_resource_or_loader_side_effects(self) -> None:
        before = dict(os.environ)
        with (
            patch.object(
                self.native,
                "_resolve_bundled_paths",
                side_effect=AssertionError("must not resolve"),
            ),
            patch.object(
                self.native,
                "_import_dependency",
                side_effect=AssertionError("must not import"),
            ),
            patch.object(
                self.native,
                "_distribution_versions",
                side_effect=AssertionError("must not query metadata"),
            ),
            patch.object(
                self.native.importlib_metadata,
                "version",
                side_effect=AssertionError("must not query distribution"),
            ),
            patch.object(
                self.native,
                "_preload_cust_opapi",
                side_effect=AssertionError("must not dlopen"),
            ),
        ):
            status = self.native.ensure_native_loaded(enabled=False)
        self.assertEqual(dict(os.environ), before)
        self.assertEqual(status.state, "disabled")
        self.assertFalse(status.loaded)
        self.assertTrue(status.compatible)

    def test_load_order_and_environment_prepend(self) -> None:
        status = self.compatible_status()
        events: list[tuple[str, str]] = []

        def preload(path: str):
            events.append(("cust_opapi", path))
            return Mock(spec=ctypes.CDLL)

        def load_adapter(path: str) -> None:
            events.append(("adapter", path))

        env = {
            "ASCEND_CUSTOM_OPP_PATH": "/existing/opp",
            "LD_LIBRARY_PATH": "/existing/lib",
        }
        with (
            patch.dict(os.environ, env, clear=True),
            patch.object(
                self.native,
                "inspect_native_compatibility",
                return_value=status,
            ),
            patch.object(
                self.native,
                "_preload_cust_opapi",
                side_effect=preload,
            ),
            patch.object(
                self.native,
                "_load_torch_adapter",
                side_effect=load_adapter,
            ),
        ):
            result = self.native.ensure_native_loaded()
            self.assertEqual(
                os.environ["ASCEND_CUSTOM_OPP_PATH"].split(os.pathsep),
                [status.opp_vendor_path, "/existing/opp"],
            )
            self.assertEqual(
                os.environ["LD_LIBRARY_PATH"].split(os.pathsep),
                [
                    os.path.dirname(status.cust_opapi_path),
                    "/existing/lib",
                ],
            )

        self.assertTrue(result)
        self.assertEqual(result.state, "ready")
        self.assertEqual(
            events,
            [
                ("cust_opapi", status.cust_opapi_path),
                ("adapter", status.adapter_path),
            ],
        )

    def test_loader_is_idempotent(self) -> None:
        status = self.compatible_status()
        with (
            patch.object(
                self.native,
                "inspect_native_compatibility",
                return_value=status,
            ) as inspect,
            patch.object(
                self.native, "_preload_cust_opapi", return_value=Mock()
            ) as preload,
            patch.object(self.native, "_load_torch_adapter") as adapter,
        ):
            first = self.native.ensure_native_loaded()
            second = self.native.ensure_native_loaded()
        self.assertIs(first, second)
        inspect.assert_called_once_with(enabled=True)
        preload.assert_called_once()
        adapter.assert_called_once()

    def test_non_strict_failure_is_queryable_and_restores_paths(self) -> None:
        status = self.compatible_status()
        with (
            patch.dict(os.environ, {}, clear=True),
            patch.object(
                self.native,
                "inspect_native_compatibility",
                return_value=status,
            ),
            patch.object(
                self.native,
                "_preload_cust_opapi",
                side_effect=OSError("missing dependency"),
            ),
        ):
            result = self.native.ensure_native_loaded(strict=False)
            self.assertNotIn("ASCEND_CUSTOM_OPP_PATH", os.environ)
            self.assertNotIn("LD_LIBRARY_PATH", os.environ)
        self.assertFalse(result)
        self.assertEqual(result.state, "load_failed")
        self.assertIn("missing dependency", result.errors[-1])
        self.assertEqual(self.native.native_status(), result)

    def test_strict_failure_raises_with_status(self) -> None:
        incompatible = self.native.NativeStatus(
            enabled=True,
            loaded=False,
            compatible=False,
            state="incompatible",
            system="Darwin",
            machine="arm64",
            python_implementation="cpython",
            python_version="3.12.0",
            errors=("unsupported operating system 'Darwin'; expected Linux",),
        )
        with patch.object(
            self.native,
            "inspect_native_compatibility",
            return_value=incompatible,
        ):
            with self.assertRaises(
                self.native.NativeBootstrapError
            ) as caught:
                self.native.ensure_native_loaded(strict=True)
        self.assertEqual(caught.exception.status, incompatible)
        self.assertIn("expected Linux", str(caught.exception))

    def test_soc_family_gate_accepts_910b_variants(self) -> None:
        self.assertTrue(self.native._soc_is_supported("Ascend910B2"))
        self.assertTrue(self.native._soc_is_supported("ascend-910b-3"))
        self.assertFalse(self.native._soc_is_supported("Ascend910A"))

    def test_compatibility_report_names_platform_and_soc_errors(self) -> None:
        with (
            patch.object(
                self.native,
                "_runtime_identity",
                return_value=("Darwin", "arm64", "cpython", "3.12.4"),
            ),
            patch.object(self.native.sys, "version_info", (3, 12, 4)),
            patch.object(
                self.native,
                "_dependency_versions",
                return_value=(
                    Mock(),
                    Mock(),
                    "2.10.0+cpu",
                    "2.10.0.post2",
                ),
            ),
            patch.object(
                self.native,
                "_distribution_versions",
                return_value={
                    "vllm": "0.23.0+empty",
                    "vllm_ascend": "0.23.0rc1",
                    "torch": "2.10.0",
                    "torch_npu": "2.10.0.post2",
                },
            ),
            patch.object(
                self.native,
                "_device_soc_version",
                return_value="Ascend910A",
            ),
        ):
            status = self.native.inspect_native_compatibility(True)
        self.assertFalse(status.compatible)
        details = " ".join(status.errors)
        self.assertIn("operating system", details)
        self.assertIn("expected aarch64", details)
        self.assertIn("requires CPython 3.10", details)
        self.assertIn("Ascend910A", details)

    def test_dependency_import_failures_are_actionable(self) -> None:
        with (
            patch.object(
                self.native,
                "_runtime_identity",
                return_value=("Linux", "aarch64", "cpython", "3.10.17"),
            ),
            patch.object(self.native.sys, "version_info", (3, 10, 17)),
            patch.object(
                self.native,
                "_import_dependency",
                side_effect=ImportError("not installed"),
            ),
            patch.object(
                self.native,
                "_distribution_versions",
                return_value={
                    "vllm": "0.23.0",
                    "vllm_ascend": "0.23.0rc1",
                    "torch": "2.10.0",
                    "torch_npu": "2.10.0.post2",
                },
            ),
            patch.object(
                self.native,
                "_device_soc_version",
                return_value="Ascend910B2",
            ),
        ):
            status = self.native.inspect_native_compatibility(True)
        details = " ".join(status.errors)
        self.assertIn("install the vLLM-Ascend runtime", details)
        self.assertIn("install a CANN-compatible torch_npu", details)

    def inspect_runtime(
        self,
        *,
        distributions: dict[str, str | None] | None = None,
        torch_version: str | None = "2.10.0+cpu",
        torch_npu_version: str | None = "2.10.0.post2",
    ):
        versions = distributions or {
            "vllm": "0.23.0+empty",
            "vllm_ascend": "0.23.0rc1",
            "torch": "2.10.0",
            "torch_npu": "2.10.0.post2",
        }
        torch_module = Mock()
        torch_module.ops.load_library = Mock()
        with (
            patch.object(
                self.native,
                "_runtime_identity",
                return_value=("Linux", "aarch64", "cpython", "3.10.17"),
            ),
            patch.object(self.native.sys, "version_info", (3, 10, 17)),
            patch.object(
                self.native,
                "_artifact_errors",
                return_value=[],
            ),
            patch.object(
                self.native,
                "_read_cann_compiler_version",
                return_value="9.0.1",
            ),
            patch.object(
                self.native,
                "_distribution_versions",
                return_value=versions,
            ),
            patch.object(
                self.native,
                "_dependency_versions",
                return_value=(
                    torch_module,
                    Mock(),
                    torch_version,
                    torch_npu_version,
                ),
            ),
            patch.object(
                self.native,
                "_device_soc_version",
                return_value="Ascend910B3",
            ),
        ):
            return self.native.inspect_native_compatibility(True)

    def test_exact_supported_runtime_fingerprint_accepts_local_suffixes(
        self,
    ) -> None:
        status = self.inspect_runtime()
        self.assertTrue(status.compatible, status.errors)
        self.assertEqual(status.state, "compatible")
        self.assertEqual(status.vllm_version, "0.23.0+empty")
        self.assertEqual(status.torch_version, "2.10.0+cpu")
        self.assertEqual(
            status.runtime_fingerprint,
            ("0.23.0", "0.23.0rc1", "2.10.0", "2.10.0.post2"),
        )

    def test_mixed_runtime_fingerprint_is_rejected_fail_closed(self) -> None:
        status = self.inspect_runtime(
            distributions={
                "vllm": "0.23.0",
                "vllm_ascend": "0.22.0",
                "torch": "2.10.0",
                "torch_npu": "2.10.0.post2",
            },
        )
        self.assertFalse(status.compatible)
        self.assertIsNone(status.runtime_fingerprint)
        details = " ".join(status.errors)
        self.assertIn("unsupported complete runtime fingerprint", details)
        self.assertIn("vllm-ascend=0.22.0", details)
        self.assertIn("vllm-ascend=0.23.0rc1", details)

    def test_prerelease_and_postrelease_markers_are_not_stripped(self) -> None:
        status = self.inspect_runtime(
            distributions={
                "vllm": "0.23.0rc1+empty",
                "vllm_ascend": "0.23.0rc1.post1",
                "torch": "2.10.0",
                "torch_npu": "2.10.0.post2",
            },
        )
        self.assertFalse(status.compatible)
        details = " ".join(status.errors)
        self.assertIn("vllm=0.23.0rc1+empty", details)
        self.assertIn("vllm-ascend=0.23.0rc1.post1", details)

    def test_unknown_module_version_is_rejected(self) -> None:
        status = self.inspect_runtime(torch_version="unknown")
        self.assertFalse(status.compatible)
        details = " ".join(status.errors)
        self.assertIn("PyTorch version 'unknown'", details)
        self.assertIn("torch=unknown", details)

    def test_unknown_distribution_version_is_rejected(self) -> None:
        status = self.inspect_runtime(
            distributions={
                "vllm": "unknown",
                "vllm_ascend": "0.23.0rc1",
                "torch": "2.10.0",
                "torch_npu": "2.10.0.post2",
            },
        )
        self.assertFalse(status.compatible)
        details = " ".join(status.errors)
        self.assertIn("vLLM version 'unknown'", details)
        self.assertIn("vllm=unknown", details)

    def test_module_and_distribution_version_mismatch_is_rejected(
        self,
    ) -> None:
        status = self.inspect_runtime(
            distributions={
                "vllm": "0.23.0",
                "vllm_ascend": "0.23.0rc1",
                "torch": "2.9.0",
                "torch_npu": "2.10.0.post2",
            },
        )
        self.assertFalse(status.compatible)
        self.assertIn(
            "PyTorch module/distribution version mismatch",
            " ".join(status.errors),
        )

    def test_missing_distribution_metadata_is_rejected(self) -> None:
        def installed_version(name: str) -> str:
            if name == "vllm":
                raise self.native.importlib_metadata.PackageNotFoundError(name)
            return {
                "vllm-ascend": "0.23.0rc1",
                "torch": "2.10.0",
                "torch-npu": "2.10.0.post2",
            }[name]

        errors: list[str] = []
        with patch.object(
            self.native.importlib_metadata,
            "version",
            side_effect=installed_version,
        ):
            versions = self.native._distribution_versions(errors)
        self.assertIsNone(versions["vllm"])
        self.assertIn(
            "vLLM distribution metadata is missing",
            " ".join(errors),
        )

    def test_status_serialises_tuples_as_json_lists(self) -> None:
        value = self.compatible_status().to_dict()
        self.assertEqual(value["errors"], [])
        self.assertEqual(value["warnings"], [])
        self.assertEqual(
            value["runtime_fingerprint"],
            ["0.23.0", "0.23.0rc1", "2.10.0", "2.10.0.post2"],
        )


if __name__ == "__main__":
    unittest.main()
