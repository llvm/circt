# Build/install the ESI runtime python package.
#
# To install:
#   pip install .
# To build a wheel:
#   pip wheel .
#
# It is recommended to build with Ninja and ccache. To do so, set environment
# variables by prefixing to above invocations:
#   CMAKE_GENERATOR=Ninja CMAKE_CXX_COMPILER=clang++
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
    # Set up build dirs.
    target_dir = os.path.abspath(self.build_lib)
    cmake_build_dir = os.getenv("PYCDE_CMAKE_BUILD_DIR")
    if not cmake_build_dir:
      cmake_build_dir = os.path.abspath(
          os.path.join(target_dir, "..", "cmake_build"))
    src_dir = _thisdir

    os.makedirs(cmake_build_dir, exist_ok=True)
    cmake_cache_file = os.path.join(cmake_build_dir, "CMakeCache.txt")
    if os.path.exists(cmake_cache_file):
      os.remove(cmake_cache_file)

    # Configure the build.
    cfg = "Release"
    cmake_args = [
        "-DCMAKE_BUILD_TYPE={}".format(cfg),  # not used on MSVC, but no harm
        "-DCAPNP_PATH={}".format(os.getenv("CAPNP_PATH")),
        "-DPython_EXECUTABLE={}".format(sys.executable.replace("\\", "/")),
        "-DWHEEL_BUILD=ON",
    ]
    cxx = os.getenv("CXX")
    if cxx is not None:
      cmake_args.append("-DCMAKE_CXX_COMPILER={}".format(cxx))

    cc = os.getenv("CC")
    if cc is not None:
      cmake_args.append("-DCMAKE_C_COMPILER={}".format(cc))

    if "CIRCT_EXTRA_CMAKE_ARGS" in os.environ:
      cmake_args += os.environ["CIRCT_EXTRA_CMAKE_ARGS"].split(" ")

    subprocess.check_call(["cmake", src_dir] + cmake_args, cwd=cmake_build_dir)

    # Run the build.
    subprocess.check_call([
        "cmake",
        "--build",
        ".",
        "--parallel",
        "--target",
        "ESIRuntime",
        "esiquery",
        "ESIPythonRuntime",
        "EsiCosimDpiServer",
    ],
                          cwd=cmake_build_dir)

    # Install the runtime directly into the target directory.
    if os.path.exists(target_dir):
      shutil.rmtree(target_dir)
    subprocess.check_call([
        "cmake", "--install", ".", "--prefix",
        os.path.join(target_dir, "esiaccel"), "--component", "ESIRuntime"
    ],
                          cwd=cmake_build_dir)


class NoopBuildExtension(build_ext):

  def build_extension(self, ext):
    pass


setup(name="esiaccel",
      include_package_data=True,
      ext_modules=[
          CMakeExtension("esiaccel.esiCppAccel"),
      ],
      cmdclass={
          "build": CustomBuild,
          "built_ext": NoopBuildExtension,
          "build_py": CMakeBuild,
      },
      zip_safe=False,
      packages=find_namespace_packages(include=[
          "esiaccel",
          "esiaccel.*",
      ]))
