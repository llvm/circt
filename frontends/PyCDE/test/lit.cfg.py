# -*- Python -*-

import os
import platform
import re
import shutil
import subprocess
import tempfile
import warnings

import lit.formats
import lit.util

from lit.llvm import llvm_config
from lit.llvm.subst import ToolSubst
from lit.llvm.subst import FindTool

# Configuration file for the 'lit' test runner.

# name: The name of this test suite.
config.name = 'PyCDE'

config.test_format = lit.formats.ShTest(not llvm_config.use_lit_shell)

# suffixes: A list of file extensions to treat as test files.
config.suffixes = ['.td', '.mlir', '.ll', '.fir', '.sv', '.py']

# test_source_root: The root path where tests are located.
config.test_source_root = os.path.dirname(__file__)

# test_exec_root: The root path where tests should be run.
config.test_exec_root = os.path.join(config.circt_obj_root, 'test')

config.substitutions.append(('%PATH%', config.environment['PATH']))
config.substitutions.append(('%shlibext', config.llvm_shlib_ext))
config.substitutions.append(('%shlibdir', config.circt_shlib_dir))
config.substitutions.append(('%INC%', config.circt_include_dir))
config.substitutions.append(('%PYTHON%', config.python_executable))

llvm_config.with_system_environment(['HOME', 'INCLUDE', 'LIB', 'TMP', 'TEMP'])

llvm_config.use_default_substitutions()

# Set the timeout, if requested.
if config.timeout is not None and config.timeout != "":
  lit_config.maxIndividualTestTime = int(config.timeout)

# excludes: A list of directories to exclude from the testsuite. The 'Inputs'
# subdirectories contain auxiliary inputs for various tests in their parent
# directories.
config.excludes = [
    'Inputs', 'CMakeLists.txt', 'README.txt', 'LICENSE.txt', 'lit.cfg.py',
    'lit.local.cfg.py'
]

# test_source_root: The root path where tests are located.
config.test_source_root = os.path.dirname(__file__)

# test_exec_root: The root path where tests should be run.
config.test_exec_root = os.path.join(config.circt_obj_root, 'frontends/pycde/test')

# Tweak the PATH to include the tools dir.
llvm_config.with_environment('PATH', config.llvm_tools_dir, append_path=True)

# Tweak the PYTHONPATH to include the binary dir.
llvm_config.with_environment('PYTHONPATH', [
    os.path.join(config.llvm_obj_root, 'python'),
    os.path.join(config.circt_obj_root, 'python')
  ],
  append_path=True)
