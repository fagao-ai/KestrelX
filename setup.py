from __future__ import annotations

import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path

from setuptools import Distribution, setup
from setuptools.command.build_py import build_py
from wheel.bdist_wheel import bdist_wheel

ROOT = Path(__file__).resolve().parent
PACKAGE = "kestrel_agent_core"


def native_library_name() -> str:
    if sys.platform == "darwin":
        return "libkestrel_ffi.dylib"
    if os.name == "nt":
        return "kestrel_ffi.dll"
    return "libkestrel_ffi.so"


def cargo_target_dir() -> Path:
    value = os.environ.get("CARGO_TARGET_DIR")
    if not value:
        return ROOT / "target"
    target = Path(value)
    return target if target.is_absolute() else ROOT / target


def native_artifact() -> Path:
    profile = os.environ.get("KESTREL_PYTHON_PROFILE", "release")
    target = cargo_target_dir()
    candidates = [target / profile / native_library_name()]
    build_target = os.environ.get("CARGO_BUILD_TARGET")
    if build_target:
        candidates.insert(0, target / build_target / profile / native_library_name())
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise RuntimeError(
        f"native library not found in {candidates}; cargo build did not produce it"
    )


class BinaryDistribution(Distribution):
    def has_ext_modules(self) -> bool:
        return True


class BuildPy(build_py):
    def run(self) -> None:
        super().run()
        self.build_native()

    def build_native(self) -> None:
        profile = os.environ.get("KESTREL_PYTHON_PROFILE", "release")
        command = ["cargo", "build", "-p", "kestrel-ffi"]
        if profile == "release":
            command.insert(2, "--release")
        command.extend(["--manifest-path", str(ROOT / "Cargo.toml")])
        subprocess.run(command, cwd=ROOT, check=True)

        destination = Path(self.build_lib) / PACKAGE / "_native"
        destination.mkdir(parents=True, exist_ok=True)
        shutil.copy2(native_artifact(), destination / native_library_name())


class PlatformWheel(bdist_wheel):
    def finalize_options(self) -> None:
        super().finalize_options()
        self.root_is_pure = False

    def get_tag(self) -> tuple[str, str, str]:
        _, _, platform_tag = super().get_tag()
        configured = os.environ.get("KESTREL_WHEEL_PLATFORM")
        if configured:
            platform_tag = configured
        elif sys.platform == "darwin":
            machine = platform.machine()
            if machine == "arm64":
                platform_tag = "macosx_11_0_arm64"
            elif machine == "x86_64":
                platform_tag = "macosx_10_9_x86_64"
        return "py3", "none", platform_tag


setup(
    distclass=BinaryDistribution,
    cmdclass={"build_py": BuildPy, "bdist_wheel": PlatformWheel},
    zip_safe=False,
)
