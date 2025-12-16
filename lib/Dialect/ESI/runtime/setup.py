#  Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
#  See https://llvm.org/LICENSE.txt for license information.
#  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# Build/install the ESI runtime python package.
#
# To install:
#   pip install .
# To build a wheel:
#   pip wheel .
#
# It is recommended to build with Ninja and ccache. To do so, set environment
# variables by prefixing to above invocations:
#   CC=clang CXX=clang++
#
# On CIs, it is often advantageous to re-use/control the CMake build directory.
# This can be set with the PYCDE_CMAKE_BUILD_DIR env var.

import os
import platform
import shutil
import subprocess
import sys
import sysconfig

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
        "-GNinja",  # This build only works with Ninja on Windows.
        "-DCMAKE_BUILD_TYPE={}".format(cfg),  # not used on MSVC, but no harm
        "-DPython3_EXECUTABLE={}".format(sys.executable.replace("\\", "/")),
        "-DWHEEL_BUILD=ON",
    ]

    # Get the nanobind cmake directory from the isolated build environment.
    # This is necessary because CMake's execute_process may not properly find
    # nanobind installed in the isolated build environment.
    try:
      import nanobind
      nanobind_dir = nanobind.cmake_dir()
      cmake_args.append("-Dnanobind_DIR={}".format(
          nanobind_dir.replace("\\", "/")))
    except ImportError:
      print("Skipping nanobind directory detection, nanobind not found.")

    cxx = os.getenv("CXX")
    if cxx is not None:
      cmake_args.append("-DCMAKE_CXX_COMPILER={}".format(cxx))

    cc = os.getenv("CC")
    if cc is not None:
      cmake_args.append("-DCMAKE_C_COMPILER={}".format(cc))

    if "VCPKG_INSTALLATION_ROOT" in os.environ:
      cmake_args.append(
          f"-DCMAKE_TOOLCHAIN_FILE={os.environ['VCPKG_INSTALLATION_ROOT']}/scripts/buildsystems/vcpkg.cmake"
      )

    if "CIRCT_EXTRA_CMAKE_ARGS" in os.environ:
      cmake_args += os.environ["CIRCT_EXTRA_CMAKE_ARGS"].split(" ")

    # HACK: CMake fails to auto-detect static linked Python installations, which
    # happens to be what exists on manylinux. We detect this and give it a dummy
    # library file to reference (which is checks exists but never gets
    # used).
    if platform.system() == "Linux":
      python_libdir = sysconfig.get_config_var('LIBDIR')
      python_library = sysconfig.get_config_var('LIBRARY')
      if python_libdir and not os.path.isabs(python_library):
        python_library = os.path.join(python_libdir, python_library)
      if python_library and not os.path.exists(python_library):
        print("Detected static linked python. Faking a library for cmake.")
        fake_libdir = os.path.join(cmake_build_dir, "fake_python", "lib")
        os.makedirs(fake_libdir, exist_ok=True)
        fake_library = os.path.join(fake_libdir,
                                    sysconfig.get_config_var('LIBRARY'))
        subprocess.check_call(["ar", "q", fake_library])
        cmake_args.append("-DPython3_LIBRARY:PATH={}".format(fake_library))

    # Finally run the cmake configure.
    subprocess.check_call(["cmake", src_dir] + cmake_args, cwd=cmake_build_dir)
    print(" ".join(["cmake", src_dir] + cmake_args))

    # Run the build.
    subprocess.check_call([
        "cmake",
        "--build",
        ".",
        "--parallel",
        "--target",
        "ESIRuntime",
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
          "build_ext": NoopBuildExtension,
          "build_py": CMakeBuild,
      },
      zip_safe=False,
      packages=find_namespace_packages(include=[
          "esiaccel",
          "esiaccel.*",
      ]))
