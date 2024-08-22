#  Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
#  See https://llvm.org/LICENSE.txt for license information.
#  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from . import codegen

from pathlib import Path
import subprocess
import sys

_thisdir = Path(__file__).absolute().resolve().parent


def run_esiquery():
  """Run the esiquery executable with the same arguments as this script."""
  esiquery = _thisdir / "bin" / "esiquery"
  return subprocess.call([esiquery] + sys.argv[1:])


def run_esi_cosim():
  """Run the esi-cosim.py script with the same arguments as this script."""
  import importlib.util
  esi_cosim = _thisdir / "bin" / "esi-cosim.py"
  spec = importlib.util.spec_from_file_location("esi_cosim", esi_cosim)
  assert spec is not None
  cosim_import = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(cosim_import)
  return cosim_import.__main__(sys.argv)


def run_cppgen():
  return codegen.run()


def get_cmake_dir():
  return _thisdir / "cmake"
