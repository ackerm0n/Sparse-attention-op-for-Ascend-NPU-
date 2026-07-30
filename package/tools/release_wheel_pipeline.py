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
"""Build and clean-install a release wheel without touching upstream files.

The CANN package must already have been built and installed into an isolated
OPP staging root.  This command copies both the OPP tree and the freshly
rebuilt/stripped Torch adapter into a temporary package source tree, builds
one wheel, validates its complete payload, and exercises install/uninstall in
a temporary ``--system-site-packages`` venv.

The official ``vllm`` and ``vllm_ascend`` trees are hashed before install,
after install, and after uninstall.  Any changed path, size, mode, symlink
target, or file content fails the pipeline.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import platform
import shutil
import stat
import subprocess
import sys
import sysconfig
import tempfile
import time
import zipfile
from importlib import metadata
from pathlib import Path, PurePosixPath
from typing import Any, Iterable


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PACKAGE_ROOT = PROJECT_ROOT / "package"
OPERATOR_ROOT = PROJECT_ROOT / "operator"
ADAPTER_STEM = "triangle_paged_attention_torch"
PLUGIN_DISTRIBUTION = "vllm-ascend-trianglemix"
UPSTREAM_IMPORTS = ("vllm", "vllm_ascend")
ALLOWED_WHEEL_ROOTS = ("vllm_ascend_trianglemix",)
PORTABLE_OPP_SET_ENV = """#!/bin/bash

_trianglemix_vendor_root="$(
    cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd
)"
export ASCEND_CUSTOM_OPP_PATH="${_trianglemix_vendor_root}:${ASCEND_CUSTOM_OPP_PATH:-}"
export LD_LIBRARY_PATH="${_trianglemix_vendor_root}/op_api/lib:${LD_LIBRARY_PATH:-}"
unset _trianglemix_vendor_root
"""


class PipelineError(RuntimeError):
    """A release invariant failed."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(
            value,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _run(
    command: Iterable[str | os.PathLike[str]],
    *,
    cwd: Path,
    env: dict[str, str],
    timeout: int,
) -> subprocess.CompletedProcess[str]:
    argv = [os.fspath(item) for item in command]
    completed = subprocess.run(
        argv,
        cwd=cwd,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=timeout,
        check=False,
    )
    if completed.returncode != 0:
        raise PipelineError(
            f"command failed ({completed.returncode}): "
            f"{' '.join(argv)}\n{completed.stdout}"
        )
    return completed


def _runtime_gate() -> None:
    if platform.system() != "Linux":
        raise PipelineError("release wheels must be built on Linux")
    if platform.machine().lower() not in ("aarch64", "arm64"):
        raise PipelineError("release wheels must be built on AArch64")
    if sys.implementation.name != "cpython" or sys.version_info[:2] != (
        3,
        10,
    ):
        raise PipelineError(
            "release wheels require the target CPython 3.10 interpreter"
        )


def _require_plugin_absent() -> None:
    try:
        installed = metadata.distribution(PLUGIN_DISTRIBUTION)
    except metadata.PackageNotFoundError:
        return
    raise PipelineError(
        "base environment already contains "
        f"{PLUGIN_DISTRIBUTION} {installed.version}; use a clean stack"
    )


def _package_root(import_name: str) -> Path:
    spec = importlib.util.find_spec(import_name)
    if spec is None or not spec.submodule_search_locations:
        raise PipelineError(
            f"official package is not importable: {import_name}"
        )
    locations = [Path(item).resolve() for item in spec.submodule_search_locations]
    if len(locations) != 1 or not locations[0].is_dir():
        raise PipelineError(
            f"{import_name} must resolve to one package directory: "
            f"{locations}"
        )
    return locations[0]


def _tree_manifest(root: Path) -> dict[str, dict[str, object]]:
    result: dict[str, dict[str, object]] = {}
    for path in sorted(root.rglob("*")):
        relative = path.relative_to(root).as_posix()
        details = path.lstat()
        record: dict[str, object] = {
            "mode": stat.S_IMODE(details.st_mode),
            "size": details.st_size,
        }
        if path.is_symlink():
            record["kind"] = "symlink"
            record["target"] = os.readlink(path)
        elif path.is_file():
            record["kind"] = "file"
            record["sha256"] = _sha256(path)
        elif path.is_dir():
            record["kind"] = "directory"
        else:
            record["kind"] = "other"
        result[relative] = record
    return result


def _upstream_manifests() -> dict[str, dict[str, object]]:
    return {
        name: {
            "root": str(root),
            "entries": _tree_manifest(root),
        }
        for name in UPSTREAM_IMPORTS
        for root in (_package_root(name),)
    }


def _manifest_digest(value: object) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _require_fresh_output(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    existing = sorted(output_dir.glob("vllm_ascend_trianglemix-*.whl"))
    if existing:
        raise PipelineError(
            "refusing to overwrite existing release wheel(s): "
            + ", ".join(str(path) for path in existing)
        )


def _validate_opp_root(opp_root: Path) -> Path:
    root = opp_root.resolve()
    vendor = root / "vendors" / "trianglemix"
    required = (
        vendor / "bin" / "set_env.bash",
        vendor / "op_api" / "lib" / "libcust_opapi.so",
        vendor
        / "op_impl"
        / "ai_core"
        / "tbe"
        / "kernel"
        / "config"
        / "ascend910b"
        / "triangle_paged_sparse_attention.json",
    )
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise PipelineError(
            "isolated OPP staging root is incomplete: "
            + ", ".join(missing)
        )
    return root


def _load_checker(operator_root: Path):
    checker_path = operator_root / "tools" / "check_release_artifact.py"
    spec = importlib.util.spec_from_file_location(
        "_trianglemix_release_checker",
        checker_path,
    )
    if spec is None or spec.loader is None:
        raise PipelineError(f"cannot load release checker: {checker_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _find_adapter(adapter_root: Path) -> Path:
    candidates = sorted(adapter_root.glob(f"{ADAPTER_STEM}*.so"))
    if len(candidates) != 1:
        raise PipelineError(
            f"expected one rebuilt adapter under {adapter_root}, "
            f"found {len(candidates)}"
        )
    return candidates[0]


def _wheel_roots(path: Path) -> set[str]:
    roots: set[str] = set()
    with zipfile.ZipFile(path) as archive:
        for item in archive.infolist():
            parts = PurePosixPath(item.filename).parts
            if parts:
                roots.add(parts[0])
    return roots


def _validate_wheel_roots(path: Path) -> None:
    unexpected = sorted(
        root
        for root in _wheel_roots(path)
        if root not in ALLOWED_WHEEL_ROOTS
        and ".dist-info" not in root
    )
    if unexpected:
        raise PipelineError(
            "wheel contains unexpected top-level payloads: "
            + ", ".join(unexpected)
        )


def _venv_site_packages(python: Path, env: dict[str, str]) -> Path:
    completed = _run(
        (
            python,
            "-I",
            "-c",
            (
                "import json,sysconfig;"
                "print(json.dumps(sysconfig.get_path('purelib')))"
            ),
        ),
        cwd=python.parent,
        env=env,
        timeout=60,
    )
    candidate = Path(
        json.loads(completed.stdout.strip().splitlines()[-1])
    ).resolve()
    venv_root = python.parents[1].resolve()
    if not candidate.is_dir() or not candidate.is_relative_to(venv_root):
        raise PipelineError(
            "cannot identify temporary venv site-packages: "
            f"{candidate}"
        )
    return candidate


def _base_environment_roots() -> tuple[Path, ...]:
    """Return the exact parent-runtime roots needed by the clean venv."""
    roots = {
        Path(sysconfig.get_path("purelib")).resolve(),
        *(_package_root(name).parent for name in UPSTREAM_IMPORTS),
    }
    invalid = sorted(
        root
        for root in roots
        if not root.is_dir() or root.is_relative_to(PROJECT_ROOT)
    )
    if invalid:
        raise PipelineError(
            "invalid clean-install base environment root(s): "
            + ", ".join(str(path) for path in invalid)
        )
    return tuple(sorted(roots, key=str))


def _write_base_environment_pth(
    site_packages: Path,
    roots: Iterable[Path],
) -> Path:
    """Expose the parent stack without exposing the candidate source tree."""
    resolved = tuple(sorted({path.resolve() for path in roots}, key=str))
    invalid = [
        path
        for path in resolved
        if not path.is_dir() or path.is_relative_to(PROJECT_ROOT)
    ]
    if invalid:
        raise PipelineError(
            "refusing invalid clean-install inheritance root(s): "
            + ", ".join(str(path) for path in invalid)
        )
    pth = site_packages / "_trianglemix_base_environment.pth"
    pth.write_text(
        "".join(f"{path}\n" for path in resolved),
        encoding="utf-8",
    )
    return pth


def _installed_plugin_path(
    python: Path,
    *,
    env: dict[str, str],
) -> Path:
    code = (
        "from importlib import metadata;"
        "from pathlib import Path;"
        f"d=metadata.distribution({PLUGIN_DISTRIBUTION!r});"
        "print(Path(d.locate_file('vllm_ascend_trianglemix')).resolve())"
    )
    completed = _run(
        (python, "-I", "-c", code),
        cwd=python.parent,
        env=env,
        timeout=60,
    )
    return Path(completed.stdout.strip().splitlines()[-1]).resolve()


def _assert_plugin_removed(site_packages: Path) -> None:
    leftovers = sorted(
        [
            *site_packages.glob("vllm_ascend_trianglemix"),
            *site_packages.glob("vllm_ascend_trianglemix-*.dist-info"),
        ]
    )
    if leftovers:
        raise PipelineError(
            "wheel uninstall left plugin files behind: "
            + ", ".join(str(path) for path in leftovers)
        )


def _stage_package(
    temporary: Path,
    *,
    opp_root: Path,
    adapter: Path,
) -> Path:
    staged = temporary / "package"
    # setup.py uses the project's single canonical README.  Keep the release
    # staging layout equivalent without maintaining a second documentation
    # copy under package/.
    shutil.copy2(PROJECT_ROOT / "README.md", temporary / "README.md")
    shutil.copytree(
        PACKAGE_ROOT,
        staged,
        symlinks=True,
        ignore=shutil.ignore_patterns(
            "build",
            "dist",
            "*.egg-info",
            "__pycache__",
            "*.pyc",
        ),
    )
    native = (
        staged
        / "src"
        / "vllm_ascend_trianglemix"
        / "_native"
    )
    if native.exists():
        shutil.rmtree(native)
    native.mkdir(parents=True)
    shutil.copy2(adapter, native / adapter.name)
    staged_opp = native / "opp"
    shutil.copytree(opp_root, staged_opp, symlinks=True)
    _write_portable_opp_set_env(staged_opp)
    return staged


def _write_portable_opp_set_env(opp_root: Path) -> Path:
    """Replace CANN's install-root script with an equivalent relocatable one."""
    script = (
        opp_root
        / "vendors"
        / "trianglemix"
        / "bin"
        / "set_env.bash"
    )
    if not script.is_file():
        raise PipelineError(f"OPP environment script is missing: {script}")
    script.write_text(PORTABLE_OPP_SET_ENV, encoding="utf-8")
    script.chmod(0o755)
    return script


def _build_adapter(
    temporary: Path,
    *,
    vllm_ascend_src: Path,
    env: dict[str, str],
    timeout: int,
) -> tuple[Path, Any]:
    staged_operator = temporary / "operator"
    staged_operator.mkdir()
    shutil.copytree(
        OPERATOR_ROOT / "torch_adapter",
        staged_operator / "torch_adapter",
        symlinks=True,
        ignore=shutil.ignore_patterns(
            "build",
            "*.egg-info",
            "__pycache__",
            "*.pyc",
            f"{ADAPTER_STEM}*.so",
        ),
    )
    (staged_operator / "tools").mkdir()
    shutil.copy2(
        OPERATOR_ROOT / "tools" / "check_release_artifact.py",
        staged_operator / "tools" / "check_release_artifact.py",
    )
    build_env = dict(env)
    build_env.update(
        {
            "VLLM_ASCEND_SRC": str(vllm_ascend_src),
            "PYTHON": sys.executable,
            "SOURCE_DATE_EPOCH": build_env.get(
                "SOURCE_DATE_EPOCH",
                "0",
            ),
        }
    )
    _run(
        ("bash", staged_operator / "torch_adapter" / "build.sh"),
        cwd=staged_operator,
        env=build_env,
        timeout=timeout,
    )
    adapter = _find_adapter(staged_operator / "torch_adapter")
    checker = _load_checker(staged_operator)
    errors = checker.validate_adapter(adapter)
    if errors:
        raise PipelineError(
            "rebuilt adapter failed the release gate:\n"
            + "\n".join(errors)
        )
    return adapter, checker


def _build_wheel(
    staged_package: Path,
    *,
    temporary_output: Path,
    env: dict[str, str],
    timeout: int,
) -> Path:
    temporary_output.mkdir()
    _run(
        (
            sys.executable,
            "-m",
            "build",
            "--wheel",
            "--no-isolation",
            "--outdir",
            temporary_output,
        ),
        cwd=staged_package,
        env=env,
        timeout=timeout,
    )
    wheels = sorted(temporary_output.glob("*.whl"))
    if len(wheels) != 1:
        raise PipelineError(
            f"expected one built wheel, found {len(wheels)}"
        )
    wheel = wheels[0]
    expected_suffix = "-cp310-cp310-linux_aarch64.whl"
    if not wheel.name.endswith(expected_suffix):
        raise PipelineError(
            f"wheel tag is not {expected_suffix}: {wheel.name}"
        )
    return wheel


def _clean_install_check(
    temporary: Path,
    *,
    wheel: Path,
    smoke_script: Path,
    baseline: dict[str, dict[str, object]],
    env: dict[str, str],
    timeout: int,
) -> dict[str, object]:
    venv_root = temporary / "clean-install-venv"
    _run(
        (
            sys.executable,
            "-m",
            "venv",
            "--system-site-packages",
            venv_root,
        ),
        cwd=temporary,
        env=env,
        timeout=timeout,
    )
    venv_python = venv_root / "bin" / "python"
    venv_pip = venv_root / "bin" / "pip"
    site_packages = _venv_site_packages(venv_python, env)
    base_environment_roots = _base_environment_roots()
    _write_base_environment_pth(
        site_packages,
        base_environment_roots,
    )

    _run(
        (
            venv_pip,
            "install",
            "--ignore-installed",
            "--no-deps",
            "--no-compile",
            wheel,
        ),
        cwd=temporary,
        env=env,
        timeout=timeout,
    )
    installed_plugin = _installed_plugin_path(
        venv_python,
        env=env,
    )
    if not installed_plugin.is_relative_to(site_packages):
        raise PipelineError(
            "wheel did not install into the temporary venv: "
            f"{installed_plugin}"
        )
    after_install = _upstream_manifests()
    if after_install != baseline:
        raise PipelineError(
            "install changed vllm or vllm_ascend files"
        )

    smoke = _run(
        (venv_python, "-I", smoke_script),
        cwd=temporary,
        env=env,
        timeout=timeout,
    )
    try:
        smoke_report = json.loads(smoke.stdout.strip().splitlines()[-1])
    except (IndexError, json.JSONDecodeError) as exc:
        raise PipelineError(
            f"installed-wheel smoke did not emit JSON: {smoke.stdout}"
        ) from exc
    if not bool(smoke_report.get("ok")):
        raise PipelineError(
            "installed-wheel smoke reported failure: "
            + json.dumps(smoke_report, sort_keys=True)
        )

    _run(
        (venv_pip, "uninstall", "--yes", PLUGIN_DISTRIBUTION),
        cwd=temporary,
        env=env,
        timeout=timeout,
    )
    _assert_plugin_removed(site_packages)
    after_uninstall = _upstream_manifests()
    if after_uninstall != baseline:
        raise PipelineError(
            "uninstall changed vllm or vllm_ascend files"
        )
    return {
        "site_packages": str(site_packages),
        "base_environment_roots": [
            str(path) for path in base_environment_roots
        ],
        "installed_plugin": str(installed_plugin),
        "smoke": smoke_report,
        "manifest_after_install_sha256": _manifest_digest(
            after_install
        ),
        "manifest_after_uninstall_sha256": _manifest_digest(
            after_uninstall
        ),
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--vllm-ascend-src",
        type=Path,
        required=True,
        help="Clean source checkout matching installed vllm-ascend 0.23.0rc1.",
    )
    parser.add_argument(
        "--opp-root",
        type=Path,
        required=True,
        help=(
            "Fresh isolated OPP install root containing "
            "vendors/trianglemix."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Empty/fresh directory that receives the one release wheel.",
    )
    parser.add_argument(
        "--report",
        type=Path,
        required=True,
        help="JSON evidence file; an existing file is never overwritten.",
    )
    parser.add_argument(
        "--smoke-script",
        type=Path,
        default=PACKAGE_ROOT / "tests" / "real_env_smoke.py",
    )
    parser.add_argument("--timeout", type=int, default=1800)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    started = time.time()
    report_path = args.report.resolve()
    if report_path.exists():
        raise PipelineError(
            f"refusing to overwrite existing report: {report_path}"
        )
    if args.timeout <= 0:
        raise PipelineError("--timeout must be positive")

    _runtime_gate()
    _require_plugin_absent()
    vllm_ascend_src = args.vllm_ascend_src.resolve()
    if not (vllm_ascend_src / "vllm_ascend").is_dir():
        raise PipelineError(
            f"not a vllm-ascend source checkout: {vllm_ascend_src}"
        )
    opp_root = _validate_opp_root(args.opp_root)
    output_dir = args.output_dir.resolve()
    _require_fresh_output(output_dir)
    smoke_script = args.smoke_script.resolve()
    if not smoke_script.is_file():
        raise PipelineError(f"smoke script is missing: {smoke_script}")

    environment = dict(os.environ)
    environment.update(
        {
            "PYTHONDONTWRITEBYTECODE": "1",
            "PIP_DISABLE_PIP_VERSION_CHECK": "1",
            "VLLM_PLUGINS": "ascend",
            "VLLM_ASCEND_ENABLE_TRIANGLE_MIX": "0",
        }
    )
    baseline = _upstream_manifests()
    baseline_digest = _manifest_digest(baseline)

    with tempfile.TemporaryDirectory(
        prefix="trianglemix-release-"
    ) as temporary_name:
        temporary = Path(temporary_name)
        adapter, checker = _build_adapter(
            temporary,
            vllm_ascend_src=vllm_ascend_src,
            env=environment,
            timeout=args.timeout,
        )
        staged_package = _stage_package(
            temporary,
            opp_root=opp_root,
            adapter=adapter,
        )
        temporary_wheel = _build_wheel(
            staged_package,
            temporary_output=temporary / "dist",
            env=environment,
            timeout=args.timeout,
        )
        wheel_errors = checker.validate_wheel(temporary_wheel)
        if wheel_errors:
            raise PipelineError(
                "wheel failed the independent release gate:\n"
                + "\n".join(wheel_errors)
            )
        _validate_wheel_roots(temporary_wheel)
        install_report = _clean_install_check(
            temporary,
            wheel=temporary_wheel,
            smoke_script=smoke_script,
            baseline=baseline,
            env=environment,
            timeout=args.timeout,
        )

        final_wheel = output_dir / temporary_wheel.name
        if final_wheel.exists():
            raise PipelineError(
                f"refusing to overwrite wheel: {final_wheel}"
            )
        shutil.copy2(temporary_wheel, final_wheel)
        final_sha256 = _sha256(final_wheel)
        adapter_sha256 = _sha256(adapter)

    final_report = {
        "schema_version": 1,
        "status": "PASS",
        "wheel": {
            "path": str(final_wheel),
            "sha256": final_sha256,
            "tag": "cp310-cp310-linux_aarch64",
        },
        "adapter": {
            "sha256": adapter_sha256,
            "aarch64_et_dyn": True,
            "debug_sections": [],
            "private_build_paths": [],
        },
        "upstream_manifest_before_sha256": baseline_digest,
        "install_isolation": install_report,
        "official_packages_unchanged": True,
        "finished_unix_seconds": time.time(),
        "duration_seconds": time.time() - started,
    }
    _write_json(report_path, final_report)
    print(json.dumps(final_report, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
