# Build/install the pycde python package. Blatantly copied from npcomp.
# Note that this includes a relatively large build of LLVM (~2400 C++ files)
# and can take a considerable amount of time, especially with defaults.
# To install:
#   pip install . --use-feature=in-tree-build
# To build a wheel:
#   pip wheel . --use-feature=in-tree-build
#
# It is recommended to build with Ninja and ccache. To do so, set environment
# variables by prefixing to above invocations:
#   CMAKE_GENERATOR=Ninja CMAKE_C_COMPILER_LAUNCHER=ccache CMAKE_CXX_COMPILER_LAUNCHER=ccache
#
# On CIs, it is often advantageous to re-use/control the CMake build directory.
# This can be set with the PYCDE_CMAKE_BUILD_DIR env var.
import os
import shutil
import subprocess
import sys

from distutils.command.build import build as _build
from setuptools import find_namespace_packages, setup, Extension
from setuptools.command.build_ext import build_ext
from setuptools.command.build_py import build_py

_thisdir = os.path.abspath(os.path.dirname(__file__))


# Build phase discovery is unreliable. Just tell it what phases to run.
class CustomBuild(_build):

  def run(self):
    self.run_command("build_py")
    self.run_command("build_ext")
    self.run_command("build_scripts")


class CMakeExtension(Extension):

  def __init__(self, name, sourcedir=""):
    Extension.__init__(self, name, sources=[])
    self.sourcedir = os.path.abspath(sourcedir)


class CMakeBuild(build_py):

  def run(self):
    target_dir = self.build_lib
    cmake_build_dir = os.getenv("PYCDE_CMAKE_BUILD_DIR")
    if not cmake_build_dir:
      cmake_build_dir = os.path.join(target_dir, "..", "cmake_build")
    cmake_install_dir = os.path.join(target_dir, "..", "cmake_install")
    circt_dir = os.path.abspath(
        os.environ.get("CIRCT_DIRECTORY", os.path.join(_thisdir, "..", "..")))
    src_dir = os.path.abspath(os.path.join(circt_dir, "llvm", "llvm"))
    cfg = "Release"
    if "BUILD_TYPE" in os.environ:
      cfg = os.environ["BUILD_TYPE"]
    cmake_args = [
        "-Wno-dev",
        "-GNinja",
        "-DCMAKE_INSTALL_PREFIX={}".format(os.path.abspath(cmake_install_dir)),
        "-DPython3_EXECUTABLE={}".format(sys.executable.replace("\\", "/")),
        "-DCMAKE_BUILD_TYPE={}".format(cfg),  # not used on MSVC, but no harm
        "-DLLVM_ENABLE_ASSERTIONS=ON",
        "-DLLVM_ENABLE_PROJECTS=mlir",
        "-DMLIR_ENABLE_BINDINGS_PYTHON=ON",
        "-DLLVM_TARGETS_TO_BUILD=host",
        "-DCIRCT_BINDINGS_PYTHON_ENABLED=ON",
        "-DCIRCT_ENABLE_FRONTENDS=PyCDE",
        "-DLLVM_EXTERNAL_PROJECTS=circt",
        "-DLLVM_EXTERNAL_CIRCT_SOURCE_DIR={}".format(circt_dir),
    ]
    # ESI runtime not currently supported on Windows.
    if os.name == "nt":
      cmake_args += ["-DESI_RUNTIME=OFF"]
    else:
      cmake_args += ["-DESI_RUNTIME=ON"]
    if "COMPILER_LAUNCHER" in os.environ:
      cmake_args += [
          f"-DCMAKE_C_COMPILER_LAUNCHER={os.environ['COMPILER_LAUNCHER']}",
          f"-DCMAKE_CXX_COMPILER_LAUNCHER={os.environ['COMPILER_LAUNCHER']}"
      ]
    if "CC" in os.environ:
      cmake_args += [f"-DCMAKE_C_COMPILER={os.environ['CC']}"]
    if "CXX" in os.environ:
      cmake_args += [f"-DCMAKE_CXX_COMPILER={os.environ['CXX']}"]
    if "CIRCT_EXTRA_CMAKE_ARGS" in os.environ:
      cmake_args += os.environ["CIRCT_EXTRA_CMAKE_ARGS"].split(" ")
    if "VCPKG_INSTALLATION_ROOT" in os.environ:
      cmake_args += [
          f"-DCMAKE_TOOLCHAIN_FILE={os.environ['VCPKG_INSTALLATION_ROOT']}/scripts/buildsystems/vcpkg.cmake"
      ]
    build_args = []
    build_parallelism = os.getenv("CMAKE_PARALLELISM")
    if build_parallelism:
      build_args.append(f"--parallel {build_parallelism}")
    else:
      build_args.append("--parallel")
    os.makedirs(cmake_build_dir, exist_ok=True)
    if os.path.exists(cmake_install_dir):
      shutil.rmtree(cmake_install_dir)
    cmake_cache_file = os.path.join(cmake_build_dir, "CMakeCache.txt")
    if os.path.exists(cmake_cache_file):
      os.remove(cmake_cache_file)
    print(f"Running cmake with args: {cmake_args}", file=sys.stderr)
    subprocess.check_call(["echo", "Running: cmake", src_dir] + cmake_args)
    subprocess.check_call(["cmake", src_dir] + cmake_args, cwd=cmake_build_dir)
    targets = ["check-pycde"]
    if "RUN_TESTS" in os.environ and os.environ["RUN_TESTS"] != "false":
      # The pycde integration tests test both PyCDE and the ESIRuntime so
      # failure shouldn't gate publishing PyCDE.
      # targets.append("check-pycde-integration")
      targets.append("check-circt")
    subprocess.check_call([
        "cmake",
        "--build",
        ".",
        "--target",
    ] + targets + build_args,
                          cwd=cmake_build_dir)
    install_cmd = ["cmake", "--build", ".", "--target", "install-PyCDE"]
    subprocess.check_call(install_cmd + build_args, cwd=cmake_build_dir)
    shutil.copytree(os.path.join(cmake_install_dir, "python_packages"),
                    target_dir,
                    symlinks=False,
                    dirs_exist_ok=True)


class NoopBuildExtension(build_ext):

  def build_extension(self, ext):
    pass


setup(name="pycde",
      include_package_data=True,
      ext_modules=[
          CMakeExtension("pycde.circt._mlir_libs._mlir"),
          CMakeExtension("pycde.circt._mlir_libs._circt"),
      ],
      cmdclass={
          "build": CustomBuild,
          "built_ext": NoopBuildExtension,
          "build_py": CMakeBuild,
      },
      zip_safe=False,
      packages=find_namespace_packages(include=[
          "pycde",
          "pycde.*",
      ]))
