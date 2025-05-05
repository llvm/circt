# -*- Python -*-

import os
import platform
import re
import shutil
import subprocess
import tempfile

import lit.formats
import lit.util

from lit.llvm import llvm_config
from lit.llvm.subst import ToolSubst
from lit.llvm.subst import FindTool

# Configuration file for the 'lit' test runner.

# name: The name of this test suite.
config.name = 'CIRCT'

config.test_format = lit.formats.ShTest(not llvm_config.use_lit_shell)

# suffixes: A list of file extensions to treat as test files.
config.suffixes = ['.td', '.mlir', '.ll', '.fir', '.sv', '.test']

# test_source_root: The root path where tests are located.
config.test_source_root = os.path.dirname(__file__)

# test_exec_root: The root path where tests should be run.
config.test_exec_root = os.path.join(config.circt_obj_root, 'test')

config.substitutions.append(('%PATH%', config.environment['PATH']))
config.substitutions.append(('%shlibext', config.llvm_shlib_ext))
config.substitutions.append(('%shlibdir', config.circt_shlib_dir))

llvm_config.with_system_environment(['HOME', 'INCLUDE', 'LIB', 'TMP', 'TEMP'])

llvm_config.use_default_substitutions()

# excludes: A list of directories to exclude from the testsuite. The 'Inputs'
# subdirectories contain auxiliary inputs for various tests in their parent
# directories.
config.excludes = ['Inputs', 'CMakeLists.txt', 'README.txt', 'LICENSE.txt']

# test_source_root: The root path where tests are located.
config.test_source_root = os.path.dirname(__file__)

# test_exec_root: The root path where tests should be run.
config.test_exec_root = os.path.join(config.circt_obj_root, 'test')

# Tweak the PATH to include the tools dir.
llvm_config.with_environment('PATH', config.llvm_tools_dir, append_path=True)
llvm_config.with_environment('PATH', config.mlir_tools_dir, append_path=True)
llvm_config.with_environment('PATH', config.circt_tools_dir, append_path=True)

tool_dirs = [
    config.circt_tools_dir, config.mlir_tools_dir, config.llvm_tools_dir
]
tools = [
    'arcilator', 'circt-as', 'circt-capi-ir-test', 'circt-capi-om-test',
    'circt-capi-firrtl-test', 'circt-capi-firtool-test',
    'circt-capi-rtg-pipelines-test', 'circt-capi-rtg-test',
    'circt-capi-rtgtest-test', 'circt-dis', 'circt-lec', 'circt-reduce',
    'circt-synth', 'circt-test', 'circt-translate', 'firtool', 'hlstool',
    'om-linker', 'kanagawatool'
]

if "CIRCT_OPT_CHECK_IR_ROUNDTRIP" in os.environ:
  tools.extend([
      ToolSubst("circt-opt", "circt-opt --verify-roundtrip",
                unresolved="fatal"),
  ])
else:
  tools.extend(["circt-opt"])

# Enable Verilator if it has been detected.
if config.verilator_path != "":
  tool_dirs.append(os.path.dirname(config.verilator_path))
  tools.append('verilator')
  config.available_features.add('verilator')

if config.zlib == "1":
  config.available_features.add('zlib')

# Enable tests for schedulers relying on an external solver from OR-Tools.
if config.scheduling_or_tools != "":
  config.available_features.add('or-tools')

# Add circt-verilog if the Slang frontend is enabled.
if config.slang_frontend_enabled:
  config.available_features.add('slang')
  tools.append('circt-verilog')
  tools.append('circt-verilog-lsp-server')

llvm_config.add_tool_substitutions(tools, tool_dirs)
